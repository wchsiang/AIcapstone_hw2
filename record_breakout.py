import os
import torch
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque, defaultdict
from breakout import DQN  # Make sure DQN model is available here

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

def evaluate_checkpoint(path, episodes=5):
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = env.action_space.n

    model = DQN((4, 84, 84), num_actions).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    total_reward = 0
    for _ in range(episodes):
        obs = env.reset()[0]
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
        state, stacked_frames = stack_frames(stacked_frames, obs, True)
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state, stacked_frames = stack_frames(stacked_frames, next_obs, False)
            episode_reward += reward

        total_reward += episode_reward

    env.close()
    return total_reward / episodes

def evaluate_and_record(model_path, output_video, max_steps=1800):
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_actions = env.action_space.n
    model = DQN((4, 84, 84), num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs = env.reset()[0]
    stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
    state, stacked_frames = stack_frames(stacked_frames, obs, True)

    frames = []
    for _ in range(max_steps):
        frame = env.render()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        with torch.no_grad():
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32).to(device)
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            break

        state, stacked_frames = stack_frames(stacked_frames, next_obs, False)

    env.close()

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_video}")

def plot_checkpoints():
    origin_dict = defaultdict(list)
    shaped_dict = defaultdict(list)

    for file in sorted(os.listdir("checkpoints_origin")):
        print(file)
        if file.endswith(".pth"):
            step = int(file.split("step")[1].split(".pth")[0])
            path = os.path.join("checkpoints_origin", file)
            reward = evaluate_checkpoint(path)
            origin_dict[step].append(reward)

    for file in sorted(os.listdir("checkpoints_shaped")):
        print(file)
        if file.endswith(".pth"):
            step = int(file.split("step")[1].split(".pth")[0])
            path = os.path.join("checkpoints_shaped", file)
            reward = evaluate_checkpoint(path)
            shaped_dict[step].append(reward)

    steps = sorted(set(origin_dict.keys()) | set(shaped_dict.keys()))
    origin_avg = [np.mean(origin_dict[step]) if step in origin_dict else None for step in steps]
    shaped_avg = [np.mean(shaped_dict[step]) if step in shaped_dict else None for step in steps]

    plt.plot(steps, origin_avg, label="Original")
    plt.plot(steps, shaped_avg, label="Shaped")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title("Checkpoint Evaluation (Averaged)")
    plt.grid()
    plt.savefig("checkpoint_comparison.png")
    plt.show()

if __name__ == "__main__":
    evaluate_and_record(model_path="checkpoints_origin/dqn_breakout_step2100000.pth", output_video="breakout_eval_origin.mp4")
    evaluate_and_record(model_path="checkpoints_shaped/dqn_breakout_step2100000.pth", output_video="breakout_eval_shaped.mp4")
    # plot_checkpoints()
