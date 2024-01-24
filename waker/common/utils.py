import numpy as np
import common
import re
import envs
import gym

CVARS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

def get_dict_key(np_array):
  """ Convert numpy array of env params to string that can be used as dict key."""
  return np.array2string(np.round(np_array.astype(float), decimals=4), formatter={'float_kind':lambda x: "%.4f" % x})

def get_per_eval_fn(logger, domain, config, train_replay, eval_replay):
    def per_eval(eps, mode, task="", name=""):
        mode_name = mode
        if task != "":
            mode_name += "_" + task
        if name != "":
            mode_name += "_" + name

        lengths = np.array([len(ep['reward']) - 1 for ep in eps])
        if task != "":
            idx = common.DOMAIN_TASK_IDS[domain].index(task)
            rets = np.array([float(ep['reward'][:, idx].astype(np.float64).sum()) for ep in eps])
        else:
            rets = np.array([float(ep['reward'].astype(np.float64).sum()) for ep in eps])
        print(f'{mode_name} of {lengths.shape[0]} episodes has avg. {np.mean(lengths)} steps and avg. return {np.mean(rets):.2f}.')

        logger.scalar(f'{mode_name}/{mode_name}_mean_return', np.mean(rets))
        logger.scalar(f'{mode_name}/{mode_name}_max_return', np.max(rets))
        logger.scalar(f'{mode_name}/{mode_name}_min_return', np.min(rets))
        logger.scalar(f'{mode_name}/{mode_name}_std_return', np.std(rets))

        logger.scalar(f'{mode_name}/{mode_name}_mean_length', np.mean(lengths))
        logger.scalar(f'{mode_name}/{mode_name}_max_length', np.max(lengths))
        logger.scalar(f'{mode_name}/{mode_name}_min_length', np.min(lengths))
        logger.scalar(f'{mode_name}/{mode_name}_std_length', np.std(lengths))

        rets = np.sort(rets)
        for cvar in CVARS:
            if cvar * lengths.shape[0] < 1:
                continue
            num_vals = int(np.round(cvar * lengths.shape[0]))
            cvar_value = np.mean(rets[:num_vals])
            logger.scalar(f'{mode_name}/{mode_name}_cvar_{cvar}_return', np.mean(cvar_value))

            # log final task completion if available
        if "task_completion" in eps[0].keys():
            if task != "":
                idx = common.DOMAIN_TASK_IDS[domain].index(task)
                completions = np.array([ep["task_completion"][-1, idx] for ep in eps])
            else:
                completions = np.array([ep["task_completion"][-1] for ep in eps])
            print(f"Task completions: {completions}")
            completions = np.sort(completions)
            
            logger.scalar(f'{mode_name}/{mode_name}_mean_completion', np.mean(completions))
            logger.scalar(f'{mode_name}/{mode_name}_max_completion', np.max(completions))
            logger.scalar(f'{mode_name}/{mode_name}_min_completion', np.min(completions))
            logger.scalar(f'{mode_name}/{mode_name}_std_completion', np.std(completions))

            for cvar in CVARS:
                if cvar * lengths.shape[0] < 1:
                    continue
                num_vals = int(np.round(cvar * lengths.shape[0]))
                cvar_value = np.mean(completions[:num_vals])
                logger.scalar(f'{mode_name}/{mode_name}_cvar_{cvar}_completion', np.mean(cvar_value))

        if config.log_keys_video != "None":
            for key in config.log_keys_video:
                if not key == "None":
                    logger.video(f'{mode_name}_policy_{key}', eps[0][key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

    return per_eval


def get_per_episode_fn(
        logger,
        domain,
        should_video_train,
        should_video_eval,
        train_replay,
        eval_replay,
        config,
        step
    ):
    def per_episode(ep, ep_num, mode, agnt=None, env_sampler=None):
        length = len(ep['reward']) - 1
        if domain in common.DOMAIN_TASK_IDS:
            scores = {
                key: np.sum([val[idx] for val in ep['reward'][1:]]) 
                for idx, key in enumerate(common.DOMAIN_TASK_IDS[domain])
            }
            print_rews = f'{mode.title()} episode has {length} steps and returns '
            print_rews += ''.join([f"{key}: {val:.1f} " for key,val in scores.items()])
            print(print_rews)
            for key,val in scores.items():
                logger.scalar(f'{mode}_return_{key}', val)
        else:
            score = float(ep['reward'].astype(np.float64).sum())
            print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
            logger.scalar(f'{mode}/{mode}_return', score)
        logger.scalar(f'{mode}/{mode}_length', length)

        if ep_num % 50 == 0:
            if env_sampler is not None:
                env_sampler_mets = env_sampler.get_metrics()
                for key, value in env_sampler_mets.items():
                    logger.scalar(f'env_sampler/{key}', value)

            # if using procedurally generated envs
            if ("env_params" in ep.keys() and mode == "train") :
                env_params = ep["env_params"][0, :]
                env_name = get_dict_key(env_params)
            for i in range(env_params.shape[0]):
                logger.scalar(f'{mode}_env/env_param_{i}', env_params[i])

            for key, value in ep.items():
                if re.match(config.log_keys_sum, key):
                    logger.scalar(f'{mode}/sum_{mode}_{key}', ep[key].sum())
                if re.match(config.log_keys_mean, key):
                    logger.scalar(f'{mode}/mean_{mode}_{key}', ep[key].mean())
                if re.match(config.log_keys_max, key):
                    logger.scalar(f'{mode}/max_{mode}_{key}', ep[key].max(0).mean())
            should = {'train': should_video_train, 'eval': should_video_eval}[mode]
            if should(step) and not config.log_keys_video == "None":
                for key in config.log_keys_video:
                    if not key == "None":
                        logger.video(f'{mode}_policy_{key}', ep[key])
            replay = dict(train=train_replay, eval=eval_replay)[mode]
            logger.add(replay.stats, prefix=mode)
            logger.write()

    return per_episode


def make_env(mode, suite, domain, config):
    if suite == 'dmc':
        env = envs.wrappers.DMC(
            domain, config.action_repeat, config.render_size, config.dmc_camera)
        env = envs.wrappers.NormalizeAction(env)
    elif suite == 'safetygym':
        safetygym_name = "Replay-" + domain + "-v0"
        env = gym.make(safetygym_name)
        env = envs.wrappers.SafetyGymWrapper(domain, env, 
                action_repeat=config.action_repeat, 
                obs_key=config.observation,
                size=config.render_size)
        env = envs.wrappers.NormalizeAction(env)
    elif suite == 'combined':
        dmc_domain = domain.split("-")[0]
        safety_gym_domain = domain.split("-")[1]
        return make_combined_env(dmc_domain, safety_gym_domain, config)
    else:
        raise NotImplementedError(suite)
    env = envs.wrappers.TimeLimit(env, config.time_limit)
    return env

def make_combined_env(dmc_domain, safety_gym_domain, config):
    dmc_env = envs.wrappers.DMC(
        dmc_domain, config.action_repeat, config.render_size, config.dmc_camera)
    dmc_env = envs.wrappers.NormalizeAction(dmc_env)
    dmc_env = envs.wrappers.TimeLimit(dmc_env, max(config.time_limit, 1000))

    safetygym_name = "Replay-" + safety_gym_domain + "-v0"
    safety_env = gym.make(safetygym_name)
    safety_env = envs.wrappers.SafetyGymWrapper(safety_gym_domain, safety_env, 
            action_repeat=config.action_repeat, 
            obs_key=config.observation,
            size=config.render_size)
    safety_env = envs.wrappers.NormalizeAction(safety_env)
    safety_env = envs.wrappers.TimeLimit(safety_env, config.time_limit)
    env = envs.wrappers.CombinedEnvWrapper(safety_gym_env=safety_env, dmc_env=dmc_env)
    return env