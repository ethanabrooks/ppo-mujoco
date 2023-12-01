import os

import gym
import torch
from gym.spaces.box import Box

from envs.dummy_vec_env import DummyVecEnv
from envs.monitor import Monitor
from envs.subproc_vec_env import SubprocVecEnv
from envs.vec_normalize import VecNormalize


def make_env(env_id: str, seed: int, rank: int, log_dir: str, allow_early_resets: bool):
    def _thunk():
        env: gym.Env = gym.make(env_id)

        env.seed(seed + rank)

        if env.spec.max_episode_steps is not None:
            env = TimeLimitMask(env)

        env = Monitor(
            env=env,
            filename=None if log_dir is None else os.path.join(log_dir, str(rank)),
            allow_early_resets=allow_early_resets,
        )

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(
    env_name: str,
    seed: int,
    num_processes: int,
    gamma: float,
    log_dir: str,
    device: torch.device,
    allow_early_resets: bool,
    dummy_vec_env: bool,
):
    envs = [
        make_env(
            env_id=env_name,
            seed=seed,
            rank=i,
            log_dir=log_dir,
            allow_early_resets=allow_early_resets,
        )
        for i in range(num_processes)
    ]

    envs: SubprocVecEnv = (
        SubprocVecEnv.make(envs)
        if dummy_vec_env or len(envs) > 1
        else DummyVecEnv.make(envs)
    )

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, done, info = self.env.step(action)  # TODO: use truncated
        spec = self.env.spec
        if done and spec.max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env: gym.Env = None, op: list[int] = [2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        assert isinstance(self.observation_space, Box)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, ob: torch.Tensor):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(gym.Wrapper):
    def __init__(self, venv: SubprocVecEnv, device: torch.Tensor):
        """Return only every `skip`-th frame"""
        super().__init__(venv)
        self.venv = venv
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step(self, action: torch.Tensor):
        action = action.detach().cpu().numpy()
        obs, reward, done, info = self.venv.step(action)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
