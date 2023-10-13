import os
from typing import Optional

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from wandb.sdk.wandb_run import Run

from external import bench
from external.vec_env.dummy_vec_env import DummyVecEnv
from external.vec_env.subproc_vec_env import SubprocVecEnv
from external.vec_env.vec_normalize import VecNormalize as VecNormalize_
from external.vec_env.vec_video_recorder import VecVideoRecorder


def make_env(env_id: str, seed: int, rank: int, log_dir: str, allow_early_resets: bool):
    def _thunk():
        env = gym.make(env_id)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)

        env = bench.Monitor(
            env,
            None if log_dir is None else os.path.join(log_dir, str(rank)),
            allow_early_resets=allow_early_resets,
        )

        if len(env.observation_space.shape) == 3:
            raise NotImplementedError

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(
    allow_early_resets: bool,
    device: torch.device,
    dummy_vec_env: bool,
    env_name: str,
    gamma: float,
    log_dir: str,
    num_processes: int,
    run: Optional[Run],
    seed: int,
    video_name: str,
):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if dummy_vec_env or len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if run is not None:
        envs = VecVideoRecorder(
            venv=envs,
            name=video_name,
            record_video_trigger=lambda x: x % 10_000 == 0,
            run=run,
            video_length=100,
        )

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
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
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, ob):
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


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
