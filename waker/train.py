import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import waker
import env_sampling

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import dreamerv2
import common


def main():

  ##### Load config #####
  configs = yaml.safe_load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)
  print('Group: ', config.group)

  ##### Configure tensorflow #####
  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    import tensorflow.keras.mixed_precision as prec
    prec.set_global_policy(prec.Policy('mixed_float16'))

  ##### Create replay buffer #####
  train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 20,
      minlen=config.dataset.length,
      maxlen=config.dataset.length))
  step = common.Counter(train_replay.stats['total_steps'])

  ##### Create logger #####
  if config["wandb"]:
    import wandb
    wandb.init(project="waker", config=config, group=config.group, sync_tensorboard=True)
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  ##### Create functions for training and evaluation #####
  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until // config.action_repeat)
  suite, domain = config.task.split('_', 1)
 
  per_episode = common.get_per_episode_fn(
      logger,
      domain,
      should_video_train,
      should_video_eval,
      train_replay,
      eval_replay,
      config,
      step
  )

  per_eval = common.get_per_eval_fn(
    logger, domain, config, train_replay, eval_replay
  )

  ##### Create environments #####
  print('Create envs.')
  num_eval_envs = min(config.envs, config.eval_eps)
  train_envs = [common.make_env('train', suite, domain, config) for _ in range(config.envs)]
  eval_envs = [common.make_env('eval', suite, domain, config) for _ in range(num_eval_envs)]
  
  ##### Create environment sampler #####
  print('Create algorithm for environment sampling.')
  if config.env_sampler == "None" or config.env_sampler == "Random":
    env_sampler = env_sampling.Random(task=config.task)
  elif config.env_sampler == "WAKER":
    env_sampler = waker.WAKER(task=config.task, **config.env_sampler_params)
  elif config.env_sampler == "HardestEnvOracle":
    env_sampler = env_sampling.HardestEnvOracle(task=config.task)
  elif config.env_sampler == "ReweightingOracle":
    env_sampler = env_sampling.ReweightingOracle(task=config.task)
  elif config.env_sampler == "GradualExpansion":
    env_sampler = env_sampling.GradualExpansion(task=config.task)
  else:
    raise NotImplementedError

  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space

  ##### Initialise agent #####
  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    prefill_driver = common.Driver(train_envs)
    prefill_driver.on_episode(lambda ep, ep_num: per_episode(ep, ep_num, mode='train', env_sampler=env_sampler))
    prefill_driver.on_step(lambda tran, worker: step.increment())
    prefill_driver.on_step(train_replay.add_step)
    prefill_driver.on_reset(train_replay.add_step)
    random_agent = common.RandomAgent(act_space)
    prefill_driver(random_agent, steps=prefill, episodes=1)
    eval_driver = common.Driver(eval_envs)
    eval_driver.on_episode(lambda ep, ep_num: eval_replay.add_episode(ep))
    while eval_replay.stats['total_episodes'] <= 0:
      eval_driver(random_agent, episodes=1)
    prefill_driver.reset()
    eval_driver.reset()

  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = dreamerv2.Agent(config, obs_space, act_space, step, domain=domain)
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
 
  ##### Set up agent training step #####
  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        data = next(train_dataset)
        mets, seq = train_agent(data)
        [metrics[key].append(value) for key, value in mets.items()]
        if env_sampler is not None and config.expl_behavior != "greedy":
          env_sampler.update_ensemble_uncert(data, seq)
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(f'train/{name}', np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.write(fps=True)
    if env_sampler is not None:
      env_sampler.update_ensemble_uncert_changes(step=step.value)

  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep, ep_num: per_episode(ep, ep_num, mode='train', agnt=agnt, env_sampler=env_sampler))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  train_driver.on_step(train_step)

  ##### Set up agent evaluation #####
  def evaluate():
    tasks = ['']
    if domain in common.DOMAIN_TASK_IDS:
      tasks = common.DOMAIN_TASK_IDS[domain]
    for task in tasks:
      print("Evaluating task: ", task)
      eval_policy = lambda *args: agnt.policy(*args, mode='eval', task=task)  
      if config.eval_mode == "standard" or config.eval_mode == "both":
        print('Start standard evaluation of randomly sampled episode contexts.')
        eval_driver = common.Driver(eval_envs)
        eval_driver.on_call(lambda eps: per_eval(eps, mode='eval', task=task, name="uniform"))
        eval_driver.on_episode(lambda ep, ep_num: eval_replay.add_episode(ep))
        eval_driver(eval_policy, episodes=config.eval_eps, task=task)
      if config.eval_mode == "cases" or config.eval_mode == "both":
        eval_cases = train_envs[0].eval_cases(task=task)
        for name, cases in eval_cases.items():
          print(f'Start evaluation of {name} test cases.')
          eval_driver = common.Driver(eval_envs)
          eval_driver.on_call(lambda eps: per_eval(eps, mode='eval', task=task, name=str(name)))
          eval_driver.on_episode(lambda ep, ep_num: eval_replay.add_episode(ep))
          eval_env_sampler = env_sampling.ParamList(cases, name)
          eval_driver(eval_policy, episodes=len(cases), env_sampler=eval_env_sampler, task=task)

  ##### Main training loop #####
  while step < config.steps:
    logger.write()
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    logger.add(agnt.report(next(report_dataset)), prefix='train')
    evaluate()
    print('Start training.')
    train_driver(train_policy, steps=config.eval_every, env_sampler=env_sampler)
    agnt.save(str(logdir)+ '/' + 'variables' + str(step.value) + '.pkl')
  evaluate()
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass

if __name__ == '__main__':
  main()
