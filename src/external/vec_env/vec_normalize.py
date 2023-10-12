import gym
import numpy as np

from external.running_mean_std import RunningMeanStd
from external.vec_env.subproc_vec_env import SubprocVecEnv


class VecNormalize(gym.Wrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(
        self,
        venv: SubprocVecEnv,
        ob: bool = True,
        ret: bool = True,
        clipob: float = 10.0,
        cliprew: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__(venv)
        self.venv = venv
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.venv.n_processes)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action: np.ndarray):
        obs, rews, news, infos = self.venv.step(action)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        self.ret[news] = 0.0
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.venv.n_processes)
        obs = self.venv.reset()
        return self._obfilt(obs)
