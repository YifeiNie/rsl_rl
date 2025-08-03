# # Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

# import torch
# import torch.nn as nn
# from torch.distributions import Normal

# from rsl_rl.networks import MLP, EmpiricalNormalization, EmpiricalNormalizationDict, Depth  # custom net 
# from tensordict import TensorDict
# class ActorCriticCustom(nn.Module):
#     is_recurrent = True

#     def __init__(
#         self,
#         obs,
#         obs_groups,
#         num_actions,
#         actor_obs_normalization=False,
#         critic_obs_normalization=False,
#         actor_hidden_dims=[256, 256, 256],
#         critic_hidden_dims=[256, 256, 256],
#         activation="elu",
#         init_noise_std=1.0,
#         noise_std_type: str = "scalar",
#         **kwargs,
#     ):
#         if kwargs:
#             print(
#                 "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
#                 + str([key for key in kwargs.keys()])
#             )
#         super().__init__()

#         # get the observation dimensions
#         self.obs_groups = obs_groups
#         num_actor_obs = self.get_shape_dict(obs)
#         num_critic_obs = 26
#         num_hiden_state = 192

#         # actor
#         self.actor = Depth(num_actor_obs, num_actions)
#         # actor observation normalization
#         self.actor_obs_normalization = actor_obs_normalization
#         if actor_obs_normalization:
#             self.actor_obs_normalizer = EmpiricalNormalizationDict(num_actor_obs)
#         else:
#             self.actor_obs_normalizer = torch.nn.Identity()
#         print(f"Actor Net: {self.actor}")

#         # critic
#         self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
#         # critic observation normalization
#         self.critic_obs_normalization = critic_obs_normalization
#         if critic_obs_normalization:
#             self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
#         else:
#             self.critic_obs_normalizer = torch.nn.Identity()
#         print(f"Critic Net: {self.critic}")

#         # Action noise
#         self.noise_std_type = noise_std_type
#         if self.noise_std_type == "scalar":
#             self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
#         elif self.noise_std_type == "log":
#             self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
#         else:
#             raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

#         # Action distribution (populated in update_distribution)
#         self.distribution = None
#         # disable args validation for speedup
#         Normal.set_default_validate_args(False)

#     def reset(self, dones=None):
#         self.actor.reset(dones=dones)
#         self.critic.reset(dones=dones)

#     def forward(self):
#         raise NotImplementedError

#     @property
#     def action_mean(self):
#         return self.distribution.mean

#     @property
#     def action_std(self):
#         return self.distribution.stddev

#     @property
#     def entropy(self):
#         return self.distribution.entropy().sum(dim=-1)

#     def get_hidden_states(self):

#         actor_hidden_states = self.actor.hidden_states
#         critic_hidden_states = torch.zeros_like(actor_hidden_states)
        
#         return actor_hidden_states, critic_hidden_states

#     def update_distribution(self, obs, hidden_states=None, masks=None):
#         # compute mean
#         mean = self.actor(obs, hidden_states=hidden_states, masks=masks)
#         # compute standard deviation
#         if self.noise_std_type == "scalar":
#             std = self.std.expand_as(mean)
#         elif self.noise_std_type == "log":
#             std = torch.exp(self.log_std).expand_as(mean)
#         else:
#             raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
#         # create distribution
#         self.distribution = Normal(mean, std)

#     def act(self, obs, hidden_states=None, masks=None, **kwargs):
#         obs = self.get_actor_obs(obs)
#         obs = self.actor_obs_normalizer(obs)
#         self.update_distribution(obs, hidden_states, masks)
#         return self.distribution.sample()

#     def act_inference(self, obs):
#         obs = self.actor_obs_normalizer(obs)
#         return self.actor(obs)

#     def evaluate(self, obs, **kwargs):
#         obs = self.get_critic_obs(obs)
#         obs = self.critic_obs_normalizer(obs)
#         return self.critic(obs)

#     def get_actor_obs(self, obs):
#         actor_obs = TensorDict({
#             "state": obs["state"], 
#             "depth": obs["depth"]}, batch_size=obs.shape[0]
#         )
#         return actor_obs

#     def get_critic_obs(self, obs):
#         return obs["privileged"]

#     def get_actions_log_prob(self, actions):
#         if self.noise_std_type == "scalar":
#             std = self.std.expand_as(actions)
#         elif self.noise_std_type == "log":
#             std = torch.exp(self.log_std).expand_as(actions)
#         else:
#             raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
#         # create distribution
#         self.distribution = Normal(actions, std)
#         return self.distribution.log_prob(actions).sum(dim=-1)
    
#     def get_shape_dict(self, tensor_dict: dict[str, torch.Tensor]) -> dict[str, tuple[int]]:
#         return {k: tuple(v.shape) for k, v in tensor_dict.items()}

#     def update_normalization(self, obs):
#         if self.actor_obs_normalization:
#             self.actor_obs_normalizer.update(obs)
#         if self.critic_obs_normalization:
#             self.critic_obs_normalizer.update(obs)

#     def load_state_dict(self, state_dict, strict=True):
#         """Load the parameters of the actor-critic model.

#         Args:
#             state_dict (dict): State dictionary of the model.
#             strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
#                            module's state_dict() function.

#         Returns:
#             bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
#                   `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
#         """

#         super().load_state_dict(state_dict, strict=strict)
#         return True  # training resumes


# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization, EmpiricalNormalizationDict, Actor_net, Critic_net  # custom net 
from tensordict import TensorDict
class ActorCriticCustom(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = self.get_shape_dict(obs)
        num_critic_obs = self.get_shape_dict(obs)
        num_hiden_state = 192

        # actor
        self.actor = Actor_net(num_actor_obs, num_actions)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalizationDict(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor Net: {self.actor}")

        # critic
        self.critic = Critic_net(num_critic_obs, 1)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalizationDict(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic Net: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(torch.tensor(init_noise_std) * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(torch.tensor(init_noise_std) * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        self.actor.reset(dones=dones)
        self.critic.reset(dones=dones)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        # compute mean
        mean = self.actor(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs):
        actor_obs = TensorDict({
            "state": obs["state"], 
            "depth": obs["depth"]}, batch_size=obs.shape[0]
        )
        return actor_obs

    def get_critic_obs(self, obs):
        actor_obs = TensorDict({
            "state": obs["state"], 
            "depth": obs["depth"],
            "privileged": obs["privileged"]}, batch_size=obs.shape[0]
        )
        return actor_obs

    def get_actions_log_prob(self, actions):
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(actions)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(actions)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(actions, std)
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_shape_dict(self, tensor_dict: dict[str, torch.Tensor]) -> dict[str, tuple[int]]:
        return {k: tuple(v.shape) for k, v in tensor_dict.items()}

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(obs)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
