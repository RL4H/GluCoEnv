import torch
import torch.nn as nn
import torch.nn.functional as F
from glucoenv.agent.utils.utils import NormedLinear
torch.set_default_dtype(torch.float32)


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.n_features = args.n_features
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.bidirectional = args.bidirectional
        self.directions = args.rnn_directions
        self.LSTM = nn.LSTM(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=self.n_layers,
                            batch_first=True, bidirectional=self.bidirectional)  # (seq_len, batch, input_size)

    def forward(self, s):
        output, (hid, cell) = self.LSTM(s)
        lstm_output = hid.view(hid.size(1), -1)  # => batch , layers * hid
        return lstm_output


class ActionModule(nn.Module):
    def __init__(self, args, device):
        super(ActionModule, self).__init__()
        self.device = device
        self.args = args
        self.output = args.n_action
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.directions = args.rnn_directions
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.normalDistribution = torch.distributions.Normal

    def forward(self, extract_states):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        mu = torch.tanh(self.mu(fc_output))
        sigma = torch.sigmoid(self.sigma(fc_output) + 1e-5)  # * 0.66
        batch_size = sigma.shape[0]
        z = self.normalDistribution(torch.zeros(batch_size, 1, device=self.device),
                                    torch.ones(batch_size, 1, device=self.device)).sample()
        action = mu + sigma * z
        action = torch.clamp(action, -1, 1)
        dst = self.normalDistribution(mu, sigma)
        log_prob = dst.log_prob(action)
        return mu, sigma, action, log_prob


class ValueModule(nn.Module):
    def __init__(self, args, device):
        super(ValueModule, self).__init__()
        self.device = device
        self.output = args.n_action
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.directions = args.rnn_directions
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.value = NormedLinear(self.last_hidden, self.output, scale=0.1)

    def forward(self, extract_states):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        value = (self.value(fc_output))
        return value


class ActorNetwork(nn.Module):
    def __init__(self, args, device):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.args = args
        self.FeatureExtractor = FeatureExtractor(args)
        self.ActionModule = ActionModule(args, self.device)
        self.distribution = torch.distributions.Normal

    def forward(self, s):
        extract_states = self.FeatureExtractor.forward(s)
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states)
        return mu, sigma, action, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, args, device):
        super(CriticNetwork, self).__init__()
        self.FeatureExtractor = FeatureExtractor(args)
        self.ValueModule = ValueModule(args, device)

    def forward(self, s):
        extract_states = self.FeatureExtractor.forward(s)
        value = self.ValueModule.forward(extract_states)
        return value


# noinspection PyCallingNonCallable
class ActorCritic(nn.Module):
    def __init__(self, args):
        super(ActorCritic, self).__init__()
        self.args = args
        self.device = args.device
        self.env_device = args.env_device
        self.experiment_dir = args.experiment_dir
        self.Actor = ActorNetwork(args, self.device)
        self.Critic = CriticNetwork(args, self.device)
        self.distribution = torch.distributions.Normal
        self.is_testing_worker = False

    def get_action(self, s):
        _, _, actions, log_probs = self.Actor(s.to(self.device))
        values = self.Critic(s.to(self.device))
        return actions.detach().to(self.env_device), values.detach().to(self.env_device), log_probs.detach().to(self.env_device)

    def get_final_value(self, s):
        return self.Critic(s.to(self.device)).detach().to(self.env_device)

    def evaluate_actor(self, state, action):  # evaluate batch
        action_mean, action_std, _, _ = self.Actor(state)
        dist = self.distribution(action_mean, action_std)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def evaluate_critic(self, state):  # evaluate batch
        state_value = self.Critic(state)
        return torch.squeeze(state_value)  # todo is this redundant

    def save(self, episode):
        actor_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Actor.pth'
        critic_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Critic.pth'
        torch.save(self.Actor, actor_path)
        torch.save(self.Critic, critic_path)

