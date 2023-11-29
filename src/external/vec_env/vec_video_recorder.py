import os
from pathlib import Path
from typing import Callable

import gym
import imageio
import numpy as np
from gym.wrappers.monitoring import video_recorder
from wandb.sdk.wandb_run import Run

from external.vec_env.subproc_vec_env import SubprocVecEnv
from wandb import Video


class VecVideoRecorder(gym.Wrapper):
    """
    Wrap VecEnv to record rendered image as mp4 video.
    """

    def __init__(
        self,
        name: str,
        record_video_trigger: Callable[[int], bool],
        run: Run,
        venv: SubprocVecEnv,
        video_length: int,
    ):
        super().__init__(venv)
        self.name = name
        self.record_video_trigger = record_video_trigger
        self.run = run
        self.venv = venv
        self.video_recorder = None

        self.file_prefix = "vecenv"
        self.file_infix = "{}".format(os.getpid())
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.path = Path(
            self.run.dir,
            "{}.video.{}.video{:06}.avi".format(
                self.file_prefix, self.file_infix, self.step_id
            ),
        )

    def reset(self):
        obs = self.venv.reset()

        self.start_video_recorder()

        return obs

    def start_video_recorder(self):
        self.close_video_recorder()

        breakpoint()
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.venv, base_path=str(self.path), metadata={"step_id": self.step_id}
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        return self.record_video_trigger(self.step_id)

    def step(self, action: np.ndarray):
        obs, rews, dones, infos = self.venv.step(action)

        self.step_id += 1
        if self.recording:
            breakpoint()
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()

        return obs, rews, dones, infos

    def close_video_recorder(self):
        if self.recording:
            self.video_recorder.close()
            if self.video_recorder.enabled:
                reader = imageio.get_reader(str(self.path))
                fps = reader.get_meta_data()["fps"]

                mp4_path = str(self.path.with_suffix(".mp4"))
                writer = imageio.get_writer(mp4_path, fps=fps)
                for im in reader:
                    writer.append_data(im)
                writer.close()
                video = Video(mp4_path)
                self.run.log(dict(video=video))
        self.recording = False
        self.recorded_frames = 0

    def close(self):
        self.venv.close(self)
        self.close_video_recorder()

    def __del__(self):
        self.close()
