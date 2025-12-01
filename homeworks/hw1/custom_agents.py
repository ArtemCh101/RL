import numpy as np
import torch
import torch.nn as nn
import random
import os
import torch.optim as optim
from collections import deque, defaultdict
import gymnasium as gym


class CustomQLAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_space = action_space
        self.bins = [np.linspace(0, 1, 10) for _ in range(100)]

    def _discretize(self, state):
        state_key = []
        for i, val in enumerate(state):
            if i < len(self.bins):
                idx = np.digitize(val, self.bins[i])
                state_key.append(idx)
        return tuple(state_key)

    def act(self, state):
        state_key = self._discretize(state)

        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()

        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, done):
        state_key = self._discretize(state)
        next_state_key = self._discretize(next_state)

        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])

        td_target = reward + self.gamma * max_next_q * (1 - done)
        td_error = td_target - current_q
        self.q_table[state_key][action] += self.alpha * td_error

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
# --------------------------------------------------------------------------------------------------------------------

def to_tensor(x, dtype=np.float32):
    if isinstance(x, torch.Tensor):
        return x

    x = np.asarray(x, dtype=dtype)

    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x)
    return x.float() if dtype == np.float32 else x.long()


def create_network(input_dim, hidden_dims, output_dim):
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(nn.ReLU())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    network = nn.Sequential(*layers)
    return network


def compute_td_target(Q_target, rewards, next_states, terminateds, gamma=0.99):
    r = to_tensor(rewards, dtype=np.float32).squeeze()
    s_next = to_tensor(next_states, dtype=np.float32)
    term = to_tensor(terminateds, dtype=bool).squeeze()

    with torch.no_grad():
        Q_sn = Q_target(s_next)
    V_sn = Q_sn.max(dim=1).values.float()

    target = r + gamma * V_sn * (~term).float()

    return target


def compute_td_loss(Q_policy, states, actions, td_target, regularizer=0.01, out_non_reduced_losses=False):
    s = to_tensor(states, dtype=np.float32)
    a = to_tensor(actions, dtype=int).long()
    target = to_tensor(td_target, dtype=np.float32)

    if a.ndim == 1:
        a = a.unsqueeze(-1)

    Q_s_a = Q_policy(s).gather(dim=1, index=a).squeeze(-1)

    td_error = target - Q_s_a
    td_losses = td_error.pow(2)

    loss = torch.mean(td_losses)
    loss += regularizer * torch.abs(Q_s_a).mean()

    per_priorities = torch.abs(td_error).detach().cpu().numpy().reshape(-1)

    if out_non_reduced_losses:
        return loss, per_priorities

    return loss, per_priorities


def symlog(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def softmax(xs, temp=1.0):
    exp_xs = np.exp((xs - np.max(xs)) / temp)
    return exp_xs / np.sum(exp_xs)


class PrioritizedReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.rng = np.random.default_rng()

    def add(self, sample, initial_priority=1.0):
        self.buffer.append((initial_priority, *sample))

    def sample_prioritized_batch(self, n_samples):
        if len(self.buffer) < n_samples:
            return None, None

        priorities = np.array([sample[0] for sample in self.buffer])

        symlogged_priorities = symlog(priorities + 1e-6)
        probabilities = softmax(symlogged_priorities)

        replace = len(self.buffer) < n_samples * 2

        indices = self.rng.choice(
            len(self.buffer),
            size=n_samples,
            p=probabilities,
            replace=replace
        )

        samples = [self.buffer[i] for i in indices]
        _, states, actions, rewards, next_states, terminateds = zip(*samples)

        batch = (
            np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(terminateds)
        )
        return batch, indices

    def update_batch(self, indices, new_priorities):
        for i, idx in enumerate(indices):
            current_sample = self.buffer[idx]
            new_sample = (new_priorities[i], *current_sample[1:])
            self.buffer[idx] = new_sample

    def get_max_priority(self):
        if not self.buffer:
            return 1.0
        return np.max([sample[0] for sample in self.buffer])


class CustomDQNAgent:
    def __init__(
            self,
            action_space,
            state_dim,
            hidden_dims=(256, 256),
            lr=1e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.9999,
            min_epsilon=0.05,
            buffer_size=50000,
            batch_size=64,
            target_update_steps=500,
            train_freq=1
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_update_steps = target_update_steps
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.current_step = 0
        self.action_space = action_space

        self.Q_policy = create_network(state_dim, hidden_dims, action_space.n)
        self.Q_target = create_network(state_dim, hidden_dims, action_space.n)
        self.Q_target.load_state_dict(self.Q_policy.state_dict())
        self.Q_target.eval()

        self.optimizer = torch.optim.Adam(self.Q_policy.parameters(), lr=lr)

        self.replay_buffer = PrioritizedReplayBuffer(maxlen=buffer_size)

    def act(self, state):
        state_tensor = to_tensor(state, dtype=np.float32)

        if random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                Q_s = self.Q_policy(state_tensor).squeeze(0)
                action = torch.argmax(Q_s).item()

        return action

    def learn(self, state, action, reward, next_state, terminated):
        self.current_step += 1

        max_priority = self.replay_buffer.get_max_priority()
        transition = (state, action, reward, next_state, terminated)
        self.replay_buffer.add(transition, initial_priority=max_priority)

        if len(self.replay_buffer.buffer) < self.batch_size or \
                self.current_step % self.train_freq != 0:
            return

        batch, indices = self.replay_buffer.sample_prioritized_batch(self.batch_size)

        if batch is None:
            return

        states, actions, rewards, next_states, terminateds = batch

        td_target = compute_td_target(self.Q_target, rewards, next_states, terminateds, gamma=self.gamma)

        self.optimizer.zero_grad()
        loss, new_priorities = compute_td_loss(self.Q_policy, states, actions, td_target, out_non_reduced_losses=True)
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_batch(indices, new_priorities)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if self.current_step % self.target_update_steps == 0:
            self.Q_target.load_state_dict(self.Q_policy.state_dict())