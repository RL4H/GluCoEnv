import csv
import time
import torch
import torch.nn as nn
from glucoenv.agent.utils.utils import scale_observations, scale_actions
from glucoenv.agent.utils.buffers import RolloutBuffer
from glucoenv.agent.ppo.models import ActorCritic


class PPO:
    def __init__(self, args, env, eval_env, device):
        self.args = args
        self.eval_env = eval_env
        self.env = env
        self.n_env = env.n_env
        self.env_device = env.device
        self.device = device
        self.args.update({'device': device, 'env_device': self.env_device, 'n_env': self.n_env})
        self.n_step = args.n_step
        self.feature_history = args.feature_history
        self.n_features = args.n_features
        self.grad_clip = args.grad_clip
        self.entropy_coef = args.entropy_coef
        self.eps_clip = args.eps_clip
        self.train_v_iters = args.n_vf_epochs
        self.train_pi_iters = args.n_pi_epochs
        self.target_kl = args.target_kl
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr
        self.batch_size = args.batch_size

        self.counter = torch.zeros(1, device=self.device)
        self.t_sim = 0.0
        self.interactions = 0.0

        self.policy = ActorCritic(args).to(self.device)
        self.value_criterion = nn.MSELoss()
        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=self.pi_lr)
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=self.vf_lr)

        self.rollout_buffer = RolloutBuffer(self.args)
        init_obs, self._last_episode_starts = self.env.reset()
        self._last_obs = scale_observations(init_obs, [self.env.sensor.min(), 0],
                                            [self.env.sensor.max(), self.args.action_scale])

        self.save_log([['policy_grad', 'value_grad', 'val_loss', 'exp_var', 'true_var', 'pi_loss',
                        'epi_rew', 't_sim', 't_update', 'interactions']], '/training_logs')
        self.training_logs = torch.zeros(10, device=self.device)

        if self.args.verbose:
            print('\nPolicy Network Parameters: {}'.format(
                sum(p.numel() for p in self.policy.Actor.parameters() if p.requires_grad)))
            print('Value Network Parameters: {}'.format(
                sum(p.numel() for p in self.policy.Critic.parameters() if p.requires_grad)))

    def save_log(self, log_name, file_name):
        with open(self.args.experiment_dir + file_name + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()

    def train_pi(self, data):
        print('Running pi update...')
        temp_loss_log = torch.zeros(1, device=self.device)
        policy_grad, pol_count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        continue_pi_training = True
        buffer_len = data['states'].shape[0]
        for i in range(self.train_pi_iters):
            start_idx, n_batch = 0, 0
            while start_idx < buffer_len:
                n_batch += 1
                end_idx = min(start_idx + self.batch_size, buffer_len)
                old_states_batch = data['states'][start_idx:end_idx, :, :]
                old_actions_batch = data['act'][start_idx:end_idx, :]
                old_logprobs_batch = data['logp'][start_idx:end_idx, :]
                advantages_batch = data['adv'][start_idx:end_idx]
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-5)

                self.optimizer_Actor.zero_grad()
                logprobs, dist_entropy = self.policy.evaluate_actor(old_states_batch, old_actions_batch)
                ratios = torch.exp(logprobs - old_logprobs_batch)
                ratios = ratios.squeeze()
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy.mean()

                # early stop: approx kl calculation
                log_ratio = logprobs - old_logprobs_batch
                approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).detach()

                if approx_kl > 1.5 * self.target_kl:
                    if self.args.verbose:
                        print('Early stop => Epoch {}, Batch {}, Approximate KL: {}.'.format(i, n_batch, approx_kl))
                    continue_pi_training = False
                    break

                if torch.isnan(policy_loss):  # for debugging only!
                    print('policy loss: {}'.format(policy_loss))
                    exit()
                temp_loss_log += policy_loss.detach()
                policy_loss.backward()
                policy_grad += torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)
                pol_count += 1
                self.optimizer_Actor.step()
                start_idx += self.batch_size
            if not continue_pi_training:
                break

        mean_pi_grad = policy_grad / pol_count if pol_count != 0 else 0
        print('The policy loss is: {}'.format(temp_loss_log))
        return mean_pi_grad, temp_loss_log

    def train_vf(self, data):
        print('Running vf update...')
        explained_var = torch.zeros(1, device=self.device)
        val_loss_log = torch.zeros(1, device=self.device)
        val_count = torch.zeros(1, device=self.device)
        value_grad = torch.zeros(1, device=self.device)
        true_var = torch.zeros(1, device=self.device)
        buffer_len = data['states'].shape[0]
        for i in range(self.train_v_iters):
            start_idx = 0
            while start_idx < buffer_len:
                end_idx = min(start_idx + self.batch_size, buffer_len)
                old_states_batch = data['states'][start_idx:end_idx, :, :]
                returns_batch = data['ret'][start_idx:end_idx]

                self.optimizer_Critic.zero_grad()
                state_values = self.policy.evaluate_critic(old_states_batch)
                value_loss = self.value_criterion(state_values, returns_batch)
                value_loss.backward()
                value_grad += torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), self.grad_clip)
                self.optimizer_Critic.step()
                val_count += 1
                start_idx += self.batch_size

                # logging.
                val_loss_log += value_loss.detach()
                y_pred = state_values.detach().flatten()
                y_true = returns_batch.flatten()
                var_y = torch.var(y_true)
                true_var += var_y
                explained_var += 1 - torch.var(y_true - y_pred) / (var_y + 1e-5)
        #print('\nvalue update: explained varience is {} true variance is {}'.format(explained_var / val_count, true_var / val_count))

        return value_grad / val_count, val_loss_log, explained_var / val_count, true_var / val_count

    def train(self, data):
        tstart_update = time.perf_counter()
        self.training_logs[0], self.training_logs[5] = self.train_pi(data)
        self.training_logs[1], self.training_logs[2], self.training_logs[3], self.training_logs[4] = self.train_vf(data)
        test_rew = self.evaluate(total_timesteps=288)
        print('the test reward: ', test_rew)
        self.training_logs[6] = test_rew
        self.training_logs[7] = self.t_sim
        self.training_logs[8] = time.perf_counter() - tstart_update
        self.training_logs[9] = self.interactions
        self.save_log([self.training_logs.detach().cpu().flatten().numpy()], '/training_logs')

    def decay_lr(self):
        self.entropy_coef = 0  # self.entropy_coef / 100
        self.pi_lr = self.pi_lr / 10
        self.vf_lr = self.vf_lr / 10
        for param_group in self.optimizer_Actor.param_groups:
            param_group['lr'] = self.pi_lr
        for param_group in self.optimizer_Critic.param_groups:
            param_group['lr'] = self.vf_lr

    def learn(self, total_timesteps=800000):
        tstart = time.perf_counter()
        tot_steps, new_obs, dones = 0, 0, 0
        while tot_steps < total_timesteps:
            tstart_sim = time.perf_counter()
            for i in range(0, self.n_step):  # run rollout
                with torch.no_grad():
                    rl_actions, values, log_probs = self.policy.get_action(self._last_obs)
                # Paper for scaling below: Non-linear Continuous Action Spaces for RL in T1D, Hettiarachchi et al.
                actions = scale_actions(rl_actions, translation_func=self.args.action_type, max=self.args.action_scale)
                new_obs, rewards, dones, infos = self.env.step(actions)
                new_obs = scale_observations(new_obs, [self.env.sensor.min(), 0],
                                                [self.env.sensor.max(), self.args.action_scale])
                self.rollout_buffer.add(self._last_obs, rl_actions, rewards, self._last_episode_starts, values, log_probs)
                self._last_obs = new_obs
                self._last_episode_starts = dones
            with torch.no_grad():
                final_val = self.policy.get_final_value(new_obs)  # todo - update done
            data = self.rollout_buffer.prepare_rollout_buffer(final_val, dones)
            tot_steps += (self.n_env * self.n_step)
            self.interactions = tot_steps
            self.t_sim = time.perf_counter() - tstart_sim
            print('\n-----------------------------------------------------')
            print('Training Progress: {:.2f}%, Elapsed time: {:.4f} minutes.'.format(min(100.00, (tot_steps/total_timesteps)*100),
                                                                                     (time.perf_counter() - tstart)/60))
            self.train(data)  # update
            print('-----------------------------------------------------')

            # reset env every 5 iterations
            self.counter += 1
            if self.counter % 5 == 0:
                #print('RESET!')
                init_obs, self._last_episode_starts = self.env.reset()
                self._last_obs = scale_observations(init_obs, [self.env.sensor.min(), 0],
                                                    [self.env.sensor.max(), self.args.action_scale])

    def evaluate(self, total_timesteps=288):
        init_obs, _ = self.eval_env.reset()
        self._last_obs_eval = scale_observations(init_obs, [self.env.sensor.min(), 0],
                                            [self.env.sensor.max(), self.args.action_scale])
        finished = torch.zeros(self.eval_env.n_env, 1, dtype=torch.bool, device=self.eval_env.device)
        test_rewards = torch.zeros(self.eval_env.n_env, 1, device=self.eval_env.device)
        for i in range(0, total_timesteps):
            with torch.no_grad():
                actions, values, log_probs = self.policy.get_action(self._last_obs_eval)
            actions = scale_actions(actions, translation_func=self.args.action_type, max=self.args.action_scale)
            new_obs, rewards, dones, infos = self.eval_env.step(actions)
            new_obs = scale_observations(new_obs, [self.env.sensor.min(), 0],
                                         [self.env.sensor.max(), self.args.action_scale])
            self._last_obs_eval = new_obs
            test_rewards = test_rewards + (rewards * torch.logical_not(finished))
            finished += dones
            if torch.sum(finished) == self.eval_env.n_env:
                break
        return torch.sum(test_rewards, dim=1).mean()






