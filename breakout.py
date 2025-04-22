import gymnasium as gym
import ale_py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
from collections import deque
from torchvision import transforms

# ==================== Utils ====================
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def stack_frames(stacked_frames, new_frame, is_new_episode):
    frame = preprocess_frame(new_frame)
    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
        for _ in range(4):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    return np.stack(stacked_frames, axis=0), stacked_frames


# ==================== Replay Buffer ====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


# ==================== Q-Network ====================
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x)


# ==================== Main Training Loop ====================
def train(shaped=False, resume_path=None, resume_step=0, total_steps=1_500_000):
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints_origin", exist_ok=True)
    os.makedirs("checkpoints_shaped", exist_ok=True)

    num_actions = env.action_space.n
    buffer = ReplayBuffer(100_000)

    policy_net = DQN((4, 84, 84), num_actions).to(device)
    target_net = DQN((4, 84, 84), num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()

    epsilon_start = 1.0
    epsilon_end = 0.05
    decay_steps = 1_000_000
    epsilon = epsilon_start

    batch_size = 32
    gamma = 0.99
    train_freq = 4
    target_update_freq = 1000
    checkpoint_freq = 100_000

    state = env.reset()[0]
    stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    episode_reward = 0
    episode = 0

    if resume_path:
        print(f"Loading checkpoint from {resume_path}...")
        policy_net.load_state_dict(torch.load(resume_path, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        start_step = resume_step
    else:
        start_step = 1

    for step in range(start_step, total_steps + 1):
        epsilon = max(epsilon_end, epsilon_start - (step / decay_steps) * (epsilon_start - epsilon_end))

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        if shaped:
            reward = np.sign(reward)
        done = terminated or truncated
        next_state, stacked_frames = stack_frames(stacked_frames, next_obs, False)

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            print(f"Step: {step}, Episode: {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}")
            state = env.reset()[0]
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            episode_reward = 0
            episode += 1

        if len(buffer) >= batch_size and step % train_freq == 0:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)

            q_values = policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states_tensor).max(1)[0]
            expected_q = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

            loss = criterion(q_values, expected_q.detach())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

        if step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % checkpoint_freq == 0:
            if shaped:
                torch.save(policy_net.state_dict(), f"checkpoints_shaped/dqn_breakout_step{step}.pth")
            else:
                torch.save(policy_net.state_dict(), f"checkpoints_origin/dqn_breakout_step{step}.pth")

    env.close()
    if shaped:
        torch.save(policy_net.state_dict(), "dqn_breakout_final_shaped.pth")
    else:
        torch.save(policy_net.state_dict(), "dqn_breakout_final_origin.pth")
    print("Training finished. Final model saved.")


if __name__ == "__main__":
    train(shaped=False)
    train(shaped=True)
