import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
env  = gym.make('MountainCar-v0',render_mode='rgb_array')

#Q-Learning Parameters
alpha=0.1 
gamma = 0.99
epsilon=0.1
episodes=10000
max_steps=200
rewards=[]
#Discretization parameters
num_states = (40,40)

state_bounds = list(zip(env.observation_space.low,env.observation_space.high))
state_bounds[1] = [-0.07,0.07]
print(state_bounds)
q_table=np.zeros(num_states+(env.action_space.n,)) # (40,40,3)


def discretize_state(state):
    discretized=[]
    for i in range(len(state)):
        scaling = (state[i]-state_bounds[i][0])/(state_bounds[i][1]-state_bounds[i][0])
        new_state = int(round((num_states[i]-1)*scaling))
        new_state = min(num_states[i]-1,max(0,new_state))
        discretized.append(new_state)
    return tuple(discretized)
def Q_learning(episodes,epsilon):
    for ep in range(episodes):
        print("ep",ep)
        state,_=env.reset()
        state = discretize_state(state)
        total_reward=0
        for step in range(max_steps):
            if np.random.uniform(0,1)<epsilon:action = env.action_space.sample() #explore action space
            else: action = np.argmax(q_table[state])
            env.render()
            next_state,reward,done,_,_ = env.step(action)
            next_state = discretize_state(next_state)
            #update Q value
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward+gamma*q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha*td_error
            
            #update state
            state=next_state
            total_reward+=reward
            if done:break
        rewards.append(total_reward)

        #Reduce epsilon to decrease exploration over time
        epsilon =max(0.01,epsilon*0.995)
        if (ep+1)%100==0: print(f"Episode: {ep + 1}, Average Reward (last 100 episodes): {np.mean(rewards[-100:])}")
    env.close() 
    # Plotting the results
    # plt.plot(rewards)
    # plt.xlabel('Episodes')
    # plt.ylabel('Total Reward')
    # plt.title('Q-Learning Training Performance')
    # plt.show()       

     

   
def run_episode(env, q_table, max_steps=200):
    state, _ = env.reset()
    state = discretize_state(state)
    total_reward = 0
    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    env.close()
    return total_reward       
Q_learning(episodes,epsilon) 
# Save Q-table to a file
np.save("q_table.npy", q_table)
print("Q-table saved to q_table.npy")

# Load Q-table from the file
loaded_q_table = np.load("q_table.npy")
print("Q-table loaded from q_table.npy")
env1 = gym.make('MountainCar-v0',render_mode="human")
total_reward = run_episode(env1, loaded_q_table)
print(f"Total reward for the episode: {total_reward}")