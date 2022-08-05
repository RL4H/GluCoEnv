import torch
import torch.nn as nn
from glucoenv.agent.utils.reward_normalizer import RewardNormalizer


class RolloutBuffer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.env_device
        self.agent_device = args.device
        self.n_env = args.n_env
        self.n_step = args.n_step
        self.n_features = args.n_features
        self.feature_history = args.feature_history
        self.shuffle_rollout = args.shuffle_rollout
        self.return_type = args.return_type
        self.gamma = args.gamma
        self.lambda_ = args.lambda_
        self.normalize_reward = args.normalize_reward
        self.reward_normaliser = RewardNormalizer(num_envs=self.n_env, cliprew=10.0, gamma=self.gamma, epsilon=1e-8,
                                                  per_env=False)
        self.observations = torch.zeros(self.n_env, self.n_step, self.feature_history, self.n_features, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(self.n_env, self.n_step, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.n_env, self.n_step, dtype=torch.float32, device=self.device)
        self.first_flag = torch.zeros(self.n_env, self.n_step + 1, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(self.n_env, self.n_step + 1, dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(self.n_env, self.n_step, dtype=torch.float32, device=self.device)
        self.ptr, self.max_size = 0, self.n_step

    def add(self, obs, act, rew, start, value, log_prob):
        self.observations[:, self.ptr, :, :] = obs
        self.actions[:, self.ptr:self.ptr + 1] = act
        self.rewards[:, self.ptr:self.ptr + 1] = rew
        self.first_flag[:, self.ptr:self.ptr + 1] = start
        self.values[:, self.ptr:self.ptr + 1] = value
        self.log_probs[:, self.ptr:self.ptr + 1] = log_prob
        self.ptr += 1

    def finish_path(self, final_v, dones):
        assert self.ptr == self.max_size
        self.values[:, self.ptr:self.ptr + 1] = final_v
        self.first_flag[:, self.ptr:self.ptr + 1] = dones
        self.ptr = 0

    def compute_gae(self):
        if self.return_type == 'discount':
            if self.normalize_reward:  # reward normalisation
                self.rewards = self.reward_normaliser(self.rewards, self.first_flag)
        if self.return_type == 'average':
            self.rewards = self.reward_normaliser(self.rewards, self.first_flag, type='average')

        vpred, reward, first = (x for x in (self.values, self.rewards, self.first_flag))
        assert first.dim() == 2
        nenv, nstep = reward.shape
        assert vpred.shape == first.shape == (nenv, nstep + 1)
        self.advantages = torch.zeros(self.n_env, self.n_step, dtype=torch.float32, device=self.device)
        lastgaelam = 0
        for t in reversed(range(nstep)):
            notlast = 1.0 - first[:, t + 1]
            nextvalue = vpred[:, t + 1]
            # notlast: whether next timestep is from the same episode
            delta = reward[:, t] + notlast * self.gamma * nextvalue - vpred[:, t]
            self.advantages[:, t] = lastgaelam = delta + notlast * self.gamma * self.lambda_ * lastgaelam
        self.v_targ = vpred[:, :-1] + self.advantages

    def prepare_rollout_buffer(self, final_v, dones):
        self.finish_path(final_v, dones)
        self.episode_mean_reward = torch.sum(self.rewards, dim=1).mean()
        self.compute_gae()
        states = self.observations.view(-1, self.feature_history, self.n_features)
        act = self.actions.view(-1, 1)
        logp = self.log_probs.view(-1, 1)
        v_targ = self.v_targ.view(-1)
        adv = self.advantages.view(-1)
        buffer_len = states.shape[0]

        if self.shuffle_rollout:
            rand_perm = torch.randperm(buffer_len)
            states = states[rand_perm, :, :]  # torch.Size([batch, n_steps, features])v
            act = act[rand_perm, :]  # torch.Size([batch, 1])
            logp = logp[rand_perm, :]  # torch.Size([batch, 1])
            v_targ = v_targ[rand_perm]  # torch.Size([batch])
            adv = adv[rand_perm]  # torch.Size([batch])

        data = dict(states=states, act=act, logp=logp, ret=v_targ, adv=adv, mean_rew=self.episode_mean_reward)
        return {k: v.detach().to(self.agent_device) for k, v in data.items()}
