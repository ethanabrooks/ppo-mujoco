from typing import Optional

import torch
import torch.nn as nn
from gym import Space
from torch.optim import Optimizer

from algo.distributions import Bernoulli, Categorical, DiagGaussian
from algo.networks import CNNBase, MLPBase, Network
from algo.storage import RolloutStorage


class Agent(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_space: Space,
        base=None,
        base_kwargs: Optional[dict] = None,
    ):
        super(Agent, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base: Network = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(
        self,
        inputs: torch.Tensor,
        rnn_hxs: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ):
        value, actor_features, rnn_hxs = self.base.forward(inputs, rnn_hxs, masks)
        dist = self.dist.forward(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, rnn_hxs

    def get_value(
        self, inputs: torch.Tensor, rnn_hxs: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(
        self,
        inputs: torch.Tensor,
        rnn_hxs: torch.Tensor,
        masks: torch.Tensor,
        action: torch.Tensor,
    ):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist.forward(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def update(
        self,
        clip_param: float,
        entropy_coef: float,
        num_mini_batch: int,
        optimizer: Optimizer,
        ppo_epoch: int,
        rollouts: RolloutStorage,
        value_loss_coef: float,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
    ):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(ppo_epoch):
            if self.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, num_mini_batch
                )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (values, action_log_probs, dist_entropy, _,) = self.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if use_clipped_value_loss:
                    value_diff: torch.Tensor = values - value_preds_batch
                    value_pred_clipped = value_preds_batch + value_diff.clamp(
                        -clip_param, clip_param
                    )
                    value_losses: torch.Tensor = values - return_batch
                    value_losses = value_losses.pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss: torch.Tensor = return_batch - values
                    value_loss = 0.5 * value_loss.pow(2).mean()

                optimizer.zero_grad()
                (
                    value_loss * value_loss_coef
                    + action_loss
                    - dist_entropy * entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = ppo_epoch * num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
