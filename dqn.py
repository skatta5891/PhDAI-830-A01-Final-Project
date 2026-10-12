import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        units = config["hidden_units"]
        self.layers = nn.Sequential(nn.Linear(6, units), nn.ReLU(),
                                    nn.Linear(units, units), nn.ReLU(), nn.Linear(units, 3))

    def forward(self, observations):
        return self.layers(observations)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.states = np.zeros((capacity, 6), dtype=np.float32)
        self.next_states = np.zeros((capacity, 6), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, terminal):
        # Wrap around and replace the oldest transition when the buffer is full.
        index = self.position
        self.states[index], self.next_states[index] = state, next_state
        self.actions[index], self.rewards[index], self.terminals[index] = action, reward, terminal
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, batch_size):
        # Sample distinct transitions instead of training on consecutive arrivals.
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return tuple(torch.from_numpy(array[indices]) for array in
                     [self.states, self.actions, self.rewards, self.next_states, self.terminals])


def greedy_action(network, observation):
    # Evaluation chooses the highest action value without random exploration.
    with torch.no_grad():
        values = network(torch.from_numpy(observation).unsqueeze(0))
    return int(values.argmax(dim=1).item())


def update_network(network, target, optimizer, batch, config):
    states, actions, rewards, next_states, terminals = batch
    predictions = network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Terminal transitions use reward alone. There is no next episode value.
        expected = rewards + config["discount"] * (1 - terminals) * target(next_states).max(dim=1).values
    loss = nn.functional.mse_loss(predictions, expected)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    optimizer.zero_grad()
    loss.backward()
    # Limit the gradient norm before changing the network weights.
    torch.nn.utils.clip_grad_norm_(network.parameters(), config["gradient_norm"])
    optimizer.step()
    return float(loss.item())
