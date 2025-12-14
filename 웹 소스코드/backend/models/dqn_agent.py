import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random


# ================== DQN 구성 요소 ==================

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        device: str = None,
        tau: float = 0.005,  # soft update 계수
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.action_dim = action_dim
        self.training_steps = 0

    def select_action(self, state, epsilon=0.0):
        """
        state: np.ndarray shape (state_dim,)
        epsilon-greedy로 액션 선택
        """
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def soft_update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # 아직 학습 안함

        batch = self.replay_buffer.sample(self.batch_size)

        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # target Q
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(dim=1, keepdim=True)[0]
            target_q = reward_batch + self.gamma * (1.0 - done_batch) * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        # target soft update
        self.soft_update_target()
        self.training_steps += 1

        return loss.item()
