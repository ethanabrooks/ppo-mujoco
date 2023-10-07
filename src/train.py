import os
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
from wandb.sdk.wandb_run import Run

from algo import utils
from algo.envs import make_vec_envs
from algo.model import Policy
from algo.ppo import PPO
from algo.storage import RolloutStorage
from algo.utils import get_vec_normalize
from evaluation import evaluate


def train(
    disable_gae: bool,
    disable_linear_lr_decay: bool,
    disable_proper_time_limits: bool,
    dummy_vec_env: bool,
    env_name: str,
    eval_interval: int,
    gae_lambda: float,
    gamma: float,
    load_path: str,
    log_interval: int,
    lr: float,
    num_processes: int,
    num_steps: int,
    num_env_steps: int,
    ppo_params: dict,
    recurrent_policy: bool,
    run: Optional[Run],
    save_dir: str,
    save_interval: int,
    seed: int,
):
    torch.manual_seed(seed)

    torch.set_num_threads(1)
    device = torch.device("cpu")

    envs = make_vec_envs(
        env_name,
        seed,
        num_processes,
        gamma,
        None,
        device,
        False,
        dummy_vec_env=dummy_vec_env,
    )

    if load_path is None:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={"recurrent": recurrent_policy},
        )
        actor_critic.to(device)
    else:
        actor_critic, ob_rms = torch.load(load_path)
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

    agent = PPO(actor_critic=actor_critic, lr=lr, **ppo_params)

    rollouts = RolloutStorage(
        num_steps,
        num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(num_env_steps) // num_steps // num_processes
    for j in range(num_updates):
        if not disable_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, j, num_updates, lr)

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            not disable_gae,
            gamma,
            gae_lambda,
            not disable_proper_time_limits,
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % save_interval == 0 or j == num_updates - 1) and save_dir != "":
            save_path = save_dir
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(
                [actor_critic, getattr(utils.get_vec_normalize(envs), "ob_rms", None)],
                os.path.join(save_path, env_name + ".pt"),
            )

        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_processes * num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                )
            )
            log = dict(
                updates=j,
                mean_reward=np.mean(episode_rewards),
                fps=int(total_num_steps / (end - start)),
            )
            if run is not None:
                run.log(log, step=total_num_steps)

        if (
            eval_interval is not None
            and len(episode_rewards) > 1
            and j % eval_interval == 0
        ):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(
                actor_critic,
                ob_rms,
                env_name,
                seed,
                num_processes,
                None,
                device,
            )
