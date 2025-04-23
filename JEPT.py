from typing import Literal

import torch
import torch.nn as nn
import yaml
from einops import rearrange, einsum

from modules import TransformerEncoder, Resampler, ActionPerceiver, FrameEncoder


class JointEmbeddingPredictiveTransformer(nn.Module):
    """
    Joint Embedding Decision Transformer class to predict actions using a perceiver.
    """

    def __init__(self, config_path, prior_model_func=None):
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.frame_encoder = FrameEncoder(model_name=config['vit_model_name'])
        self.frame_adapter = Resampler(config_path=config['frame_adapter'])
        self.prompt_resampler = Resampler(config_path=config['prompt_resampler'])
        self.action_perceiver = ActionPerceiver(config_path=config['action_perceiver'])
        self.trajectory_encoder = TransformerEncoder(config_path=config['trajectory_encoder'])
        self.decision_transformer = TransformerEncoder(config_path=config['decision_transformer'])
        self.prior_model = prior_model_func() if prior_model_func is not None else None
        self.prior_adapter = Resampler(config_path=config['prior_adapter']) if self.prior_model is not None else None
        self.context_length = config['context_length']

    def get_parameters(self):
        if self.prior_model is not None:
            traj_params = list(self.trajectory_encoder.parameters()) + list(self.prior_adapter.parameters())
            other_params = list(self.frame_adapter.parameters()) + list(self.prompt_resampler.parameters()) + \
                           list(self.action_perceiver.parameters()) + list(self.decision_transformer.parameters())
        else:
            traj_params = list(self.trajectory_encoder.parameters()) + list(self.frame_adapter.parameters())
            other_params = list(self.prompt_resampler.parameters()) + list(self.action_perceiver.parameters()) + \
                           list(self.decision_transformer.parameters())

        return traj_params, other_params

    def encode_obs(self, obs):
        """
        :param obs: the observation tensor (B, T, C, H, W)
        :return:
        """
        with torch.no_grad():
            obs_emb = self.frame_encoder(obs)

        obs_emb = self.frame_adapter(obs_emb)
        return obs_emb

    def encode_prior(self, obs):
        """
        :param obs: the observation tensor (B, T+1, C, H, W)
        :return:
        """
        assert self.prior_model is not None, "Prior model is not provided"
        self.prior_model.to(self.device)
        with torch.no_grad():
            obs_prior = self.prior_model.encode(obs)
        obs_prior = self.prior_adapter(obs_prior)
        self.prior_model.to('cpu')
        return obs_prior

    def encode_prompt(self, prompt, pad_mask=None):
        """
        :param prompt: the prompt tensor (B, M, C, H, W)
        :param pad_mask: the padding mask tensor (B, M)
        :return:
        """
        prompt_emb = self.encode_trajectory(prompt, pad_mask)  # (B, M, L, D)
        prompt_emb = rearrange(prompt_emb, 'b m l d -> b (m l) d')
        prompt_emb = self.prompt_resampler(prompt_emb, pad_mask)  # (B, M`, D)
        return prompt_emb

    def encode_trajectory(self, obs, pad_mask=None):
        """
        :param obs: the observation tensor (B, T, C, H, W)
        :param pad_mask: the padding mask tensor (B, T)
        :return:
        """
        context_length = obs.shape[1]

        if self.prior_model is not None:
            traj_emb = (self.encode_obs(obs[:, 0:1]), self.encode_prior(obs))
            traj_emb = torch.concatenate(traj_emb, dim=1)
        else:
            traj_emb = self.encode_obs(obs)

        traj_emb = rearrange(traj_emb, 'b t l d -> b (t l) d')
        traj_emb = self.trajectory_encoder(traj_emb, padding_mask=pad_mask)
        traj_emb = rearrange(traj_emb, 'b (t l) d -> b t l d', t=context_length)

        return traj_emb

    def encode_decision(self, obs, prompt_emb, traj_emb, act_mask=None, act_head=True):
        """
        :param obs: the observation tensor (B, T, C, H, W)
        :param prompt_emb: the prompt tensor (B, M, D)
        :param traj_emb: the trajectory tensor (B, T, L, D)
        :param act_mask: the action mask tensor (B, 1)
        :param act_head: whether to return the action head
        :return:
        """

        obs_emb = self.encode_obs(obs)  # (B, T, L, D)

        prompt_length = prompt_emb.shape[1]
        context_length = obs_emb.shape[1]

        dec_emb = rearrange(torch.stack((obs_emb, traj_emb), dim=2), 'b t n l d -> b (t n l) d', n=2)
        dec_emb = torch.concatenate((prompt_emb, dec_emb), dim=1)
        dec_emb = self.decision_transformer(dec_emb)

        dec_emb = rearrange(dec_emb[:, prompt_length:], 'b (t n l) d -> b t n l d', t=context_length, n=2)

        dec_emb = dec_emb[:, :, 0]
        if act_head:
            act_emb = dec_emb[:, :, 1]
            if act_mask is not None:
                act_emb = act_emb[act_mask]
            act_logit = self.action_perceiver(act_emb)
            return act_logit, dec_emb
        else:
            return dec_emb

    def forward(self, prompt, obs, act_mask, branch_name: Literal['trajectory', 'decision'] = 'decision'):
        """
        :param prompt: prompt tensor # (B, L, C, H, W)
        :param obs: observation tensor  # (B, T+1, C, H, W)
        :param act_mask: mark the data pieces requiring action prediction # (B, 1)
        :param branch_name: the name of the branch, choose from ['trajectory', 'decision']
        """
        if branch_name == 'decision':
            with torch.no_grad():
                traj_emb = self.encode_trajectory(obs)[:, 1:]
            prompt_emb = self.encode_prompt(prompt)
            obs = obs[:, :-1]
            act_logit, dec_emb = self.encode_decision(obs, prompt_emb, traj_emb, act_mask)
            return dec_emb, traj_emb, act_logit

        elif branch_name == 'trajectory':
            vis_mask = ~act_mask
            traj_emb = self.encode_trajectory(obs)[:, 1:]
            act_traj_emb = traj_emb[act_mask]
            vis_traj_emb = traj_emb[vis_mask]
            with torch.no_grad():
                prompt_emb = self.encode_prompt(prompt)
                act_prompt_emb = prompt_emb[act_mask]
                act_obs = obs[vis_mask]
                vis_prompt_emb = prompt_emb[act_mask]
                vis_obs = obs[vis_mask]
                vis_dec_emb = self.encode_decision(vis_obs, vis_prompt_emb, vis_traj_emb, act_head=False)
            act_logit, act_dec_emb = self.encode_decision(act_obs, act_prompt_emb, act_traj_emb)

            return torch.concatenate((act_dec_emb.clone().detach(), vis_dec_emb)), \
                torch.concatenate((act_traj_emb, vis_traj_emb)), \
                act_logit

        else:
            raise ValueError(f"Invalid branch name: {branch_name}")

    @torch.no_grad()
    def rollout(self, prompt, context, dec_emb_history=None, return_dec_emb=False):
        """
        :param prompt: prompt tensor (B, M, C, H, W)
        :param context: context tensor (B, T, C, H, W)
        :param dec_emb_history: history of decision embeddings (B, T-1, L, D)
        :param return_dec_emb: whether to return decision embeddings
        """

        assert len(context.shape) == 5, "Context should be a 5D tensor (B, T, C, H, W)"

        prompt_emb = self.encode_prompt(prompt)  # (B, M, D)
        context_emb = self.encode_trajectory(context)[:, 1:] if dec_emb_history is None else dec_emb_history
        context_emb = torch.concatenate((context_emb, torch.zeros_like(context_emb[:, :1])), dim=1)  # (B, T, L, D)
        dec_emb = self.encode_decision(context, prompt_emb, context_emb, act_head=False)
        context_emb[:, -1] = dec_emb[:, -1]
        act_logit, _ = self.encode_decision(context, prompt_emb, context_emb, act_head=True)

        if return_dec_emb:
            return act_logit, context_emb[:, 1:]
        else:
            return act_logit
