import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============ PPO Actor-Critic 네트워크 ============
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, x):
        logits = self.net(x)
        return logits


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        value = self.net(x)
        return value

class PPOAgent:
    def __init__(
        self, state_dim, action_dim,
        lr=3e-4, gamma=0.99, clip_eps=0.2,
        rollout_steps=2048, batch_size=64, update_epochs=10
    ):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.buffer = []  # rollout 저장용

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.actor(state_tensor)
        prob = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(prob)

        action = dist.sample()
        logprob = dist.log_prob(action)

        return int(action.item()), logprob.item()

    def store(self, transition):
        self.buffer.append(transition)

    def compute_returns_and_advantages(self, next_value):
        states, actions, rewards, dones, logprobs, values = zip(*self.buffer)
        values = list(values) + [next_value]

        returns = []
        advantages = []
        G = next_value
        A = 0
        
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * values[t+1] * (1 - dones[t])
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            A = delta + self.gamma * A * (1 - dones[t])
            returns.insert(0, G)
            advantages.insert(0, A)

        return torch.FloatTensor(returns), torch.FloatTensor(advantages)

    def update(self):
        states, actions, rewards, dones, old_logprobs, values = zip(*self.buffer)

        next_value = values[-1]
        returns, advantages = self.compute_returns_and_advantages(next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_logprobs = torch.FloatTensor(old_logprobs)

        dataset_size = len(states)

        for _ in range(self.update_epochs):
            idxs = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = idxs[start:start+self.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_logprobs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                logits = self.actor(batch_states)
                prob = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(prob)

                new_log_probs = dist.log_prob(batch_actions)

                values_pred = self.critic(batch_states).squeeze()

                # ratio
                ratio = (new_log_probs - batch_old_log_probs).exp()

                # clipped PPO objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # value loss
                critic_loss = (batch_returns - values_pred).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

