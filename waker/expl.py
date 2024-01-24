import tensorflow as tf
from tensorflow_probability import distributions as tfd
from collections import deque

import dreamerv2
import common
import numpy as np


class RandomExplore(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.act_space = act_space
    self.config = config
    self.reward = reward
    self.wm = wm
    self.actor = self.random_actor
    stoch_size = config.rssm.stoch
    if config.rssm.discrete:
      stoch_size *= config.rssm.discrete
    size = {
        'embed': 32 * config.encoder.cnn_depth,
        'stoch': stoch_size,
        'deter': config.rssm.deter,
        'feat': stoch_size + config.rssm.deter,
    }[self.config.disag_target]
    self._networks = [
        common.MLP(size, **config.expl_head)
        for _ in range(config.disag_models)]
    self.opt = common.Optimizer('expl', **config.expl_opt)
    self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
    self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  def random_actor(self, feat):
    shape = feat.shape[:-1] + self.act_space.shape
    if self.config.actor.dist == 'onehot':
      return common.OneHotDist(tf.zeros(shape))
    else:
      dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
      return tfd.Independent(dist, 1)
    
  def train(self, start, context, data):
    metrics = {}
    stoch = start['stoch']
    if self.config.rssm.discrete:
      stoch = tf.reshape(
          stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
    target = {
        'embed': context['embed'],
        'stoch': stoch,
        'deter': start['deter'],
        'feat': context['feat'],
    }[self.config.disag_target]
    inputs = context['feat']
    if self.config.disag_action_cond:
      action = tf.cast(data['action'], inputs.dtype)
      inputs = tf.concat([inputs, action], -1)
    expl_metrics, training_seq = self.wm_sequence(
        self.wm, start, data['is_terminal'], self._intr_reward)
    ens_mets = self._train_ensemble(inputs, target)
    metrics.update(ens_mets)
    metrics.update(expl_metrics)
    return None, metrics, training_seq
  
  @tf.function
  def wm_sequence(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.config.imag_horizon
    seq = world_model.imagine(self.random_actor, start, is_terminal, hor)
    reward = reward_fn(seq)
    seq['reward'], mets1 = self.rewnorm(reward)
    mets1 = {f'reward_{k}': v for k, v in mets1.items()}
    metrics.update(**mets1)
    return metrics, seq

  def _intr_reward(self, seq):
    inputs = seq['feat']
    if self.config.disag_action_cond:
      action = tf.cast(seq['action'], inputs.dtype)
      inputs = tf.concat([inputs, action], -1)
    preds = [head(inputs).mode() for head in self._networks]
    disag = tf.tensor(preds).std(0).mean(-1)
    if self.config.disag_log:
      disag = tf.math.log(disag)
    reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]
    if self.config.expl_extr_scale:
      reward += self.config.expl_extr_scale * self.extr_rewnorm(
          self.reward(seq))[0]
    return reward

  def _train_ensemble(self, inputs, targets):
    inputs = tf.cast(inputs, tf.float32)
    targets = tf.cast(targets, tf.float32)
    if self.config.disag_offset:
      targets = targets[:, self.config.disag_offset:]
      inputs = inputs[:, :-self.config.disag_offset]
    targets = tf.stop_gradient(targets)
    inputs = tf.stop_gradient(inputs)
    with tf.GradientTape() as tape:
      preds = [head(inputs) for head in self._networks]
      loss = -sum([pred.log_prob(targets).mean() for pred in preds])
    metrics = self.opt(tape, loss, self._networks)
    return metrics

class Plan2Explore(RandomExplore):

  def __init__(self, config, act_space, wm, tfstep, reward):
    super().__init__(config, act_space, wm, tfstep, reward)
    self.ac = dreamerv2.ActorCritic(config, act_space, tfstep)
    self.actor = self.ac.actor

  def train(self, start, context, data):
    metrics = {}
    stoch = start['stoch']
    if self.config.rssm.discrete:
      stoch = tf.reshape(
          stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
    target = {
        'embed': context['embed'],
        'stoch': stoch,
        'deter': start['deter'],
        'feat': context['feat'],
    }[self.config.disag_target]
    inputs = context['feat']
    if self.config.disag_action_cond:
      action = tf.cast(data['action'], inputs.dtype)
      inputs = tf.concat([inputs, action], -1)
    ac_metrics, training_seq = self.ac.train(
        self.wm, start, data['is_terminal'], self._intr_reward)
    ens_mets = self._train_ensemble(inputs, target)
    metrics.update(ens_mets)
    metrics.update(ac_metrics)
    return None, metrics, training_seq

class ModelLoss(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.config = config
    self.reward = reward
    self.wm = wm
    self.ac = dreamerv2.ActorCritic(config, act_space, tfstep)
    self.actor = self.ac.actor
    self.head = common.MLP([], **self.config.expl_head)
    self.opt = common.Optimizer('expl', **self.config.expl_opt)

  def train(self, start, context, data):
    metrics = {}
    target = tf.cast(context[self.config.expl_model_loss], tf.float32)
    with tf.GradientTape() as tape:
      loss = -self.head(context['feat']).log_prob(target).mean()
    metrics.update(self.opt(tape, loss, self.head))
    metrics.update(self.ac.train(
        self.wm, start, data['is_terminal'], self._intr_reward))
    return None, metrics

  def _intr_reward(self, seq):
    reward = self.config.expl_intr_scale * self.head(seq['feat']).mode()
    if self.config.expl_extr_scale:
      reward += self.config.expl_extr_scale * self.reward(seq)
    return reward
