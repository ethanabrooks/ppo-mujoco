import time
from collections import deque
from typing import Optional

import gym
import numpy as np

from external.bench.monitor import ResultsWriter
from external.vec_env.subproc_vec_env import SubprocVecEnv


class VecMonitor(gym.Wrapper):
    def __init__(
        self,
        venv: SubprocVecEnv,
        filename: Optional[str] = None,
        keep_buf: int = 0,
        info_keywords: tuple = (),
    ):
        super().__init__(self, venv)
        self.venv = venv
        self.eprets = None
        self.eplens = None
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(
                filename, header={"t_start": self.tstart}, extra_keys=info_keywords
            )
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.venv.n_processes, "f")
        self.eplens = np.zeros(self.venv.n_processes, "i")
        return obs

    def step(self, action: np.ndarray):
        obs, rews, dones, infos = self.venv.step(action)
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {
                    "r": ret,
                    "l": eplen,
                    "t": round(time.time() - self.tstart, 6),
                }
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info["episode"] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info
        return obs, rews, dones, newinfos
