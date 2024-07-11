import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, action_space)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

# Function to select an action
def select_action(policy, state):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    return action.item(), log_prob

# Function to update the policy
def update_policy(optimizer, log_probs, rewards, gamma):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    
    discounted_rewards = torch.FloatTensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_loss = []
    for log_prob, reward in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * reward)
    
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

# Main function to train the policy
def train_reinforce(env_name, num_episodes, gamma, learning_rate, model_path, max_steps_per_episode):
    env = gym.make(env_name)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    policy = PolicyNetwork(state_space, action_space)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path))
        print("Loaded existing model.")
        test_model(env, policy)
        return
    else:
        print("Training a new model.")

    for episode in range(num_episodes):
        state = env.reset()[0]
        log_probs = []
        rewards = []

        done = False
        for step in range(max_steps_per_episode):
            action, log_prob = select_action(policy, state)
            next_state, reward, done, _, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

            if done:
                break

        update_policy(optimizer, log_probs, rewards, gamma)
        print(f"Episode {episode + 1}/{num_episodes} finished with total reward {sum(rewards)}")

    torch.save(policy.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def test_model(env, policy):
    env = gym.make("CartPole-v1",render_mode="human")
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        env.render()
        action, _ = select_action(policy, state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    env.close()
    print(f"Test completed with total reward {total_reward}")

if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    NUM_EPISODES = 1000
    GAMMA = 0.99
    LEARNING_RATE = 0.01
    MODEL_PATH = "reinforce_cartpole.pth"
    MAX_STEPS_PER_EPISODE = 1000  # Set a limit on the maximum number of steps per episode

    train_reinforce(ENV_NAME, NUM_EPISODES, GAMMA, LEARNING_RATE, MODEL_PATH, MAX_STEPS_PER_EPISODE)
