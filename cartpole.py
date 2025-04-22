import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
from gymnasium.wrappers import RecordVideo
import os

# ------------------ DQN 結構 ------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ------------------ Agent 定義 ------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()
            with torch.no_grad():
                target_tensor = self.model(state).clone()
                target_tensor[action] = torch.tensor(float(target))  # <<<< 修正這裡
            loss = nn.MSELoss()(self.model(state), target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# ------------------ 訓練函式 ------------------
def test_cartpole(agent, episodes=10):
    env = gym.make("CartPole-v1")
    total_rewards = []

    for _ in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

    env.close()
    return np.mean(total_rewards)

def train_cartpole(shaped=False, episodes=1000):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    test_rewards = []

    for ep in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if shaped:
                x, x_dot, theta, theta_dot = next_state
                reward -= 0.1 * abs(x) + 0.1 * abs(theta)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

        agent.replay()
        agent.update_target()

        if (ep + 1) % 100 == 0:
            test_reward = test_cartpole(agent, episodes=10)
            test_rewards.append(test_reward)
            label = "Shaped" if shaped else "Original"
            print(f"[{label}] Episode {ep+1} - Test Avg Reward: {test_reward:.2f} - Epsilon: {agent.epsilon:.4f}")

    env.close()
    return test_rewards, agent

# ------------------ 錄製影片函式 ------------------
def record_cartpole_video(agent, output_dir="videos", video_length=500):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=output_dir, episode_trigger=lambda x: True)
    state = env.reset()[0]
    done = False
    step = 0
    while not done and step < video_length:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        done = terminated or truncated
        step += 1
    env.close()
    print(step)
    print(f"Video saved to: {os.path.abspath(output_dir)}")

# ------------------ 執行訓練與錄製 ------------------
rewards_original, agent_origin = train_cartpole(shaped=False)
rewards_shaped, agent_shaped = train_cartpole(shaped=True)

# record_cartpole_video(agent_origin, output_dir="videos/origin")
# record_cartpole_video(agent_shaped, output_dir="videos/shaped")

# ------------------ 繪圖 ------------------
x = np.arange(100, 100 * (len(rewards_original) + 1), 100)
plt.plot(x, rewards_original, label='Original')
plt.plot(x, rewards_shaped, label='Shaped')
plt.xlabel('Episode')
plt.ylabel('Test Avg Reward')
plt.title('Test Performance Comparison: Original vs. Shaped Rewards')
plt.legend()
plt.grid()
plt.savefig("cartpole_comparison.png")
plt.show()
