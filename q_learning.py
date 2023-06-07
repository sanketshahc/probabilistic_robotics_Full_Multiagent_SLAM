import gym
import numpy as np
import matplotlib.pyplot as plt


T = 5000  # Total number of episodes
alpha = 0.75  # Learning rate
gamma = 0.9  # Discount factor
eps = 2  # Amount of randomness in the action selection
eps_decay = 0.0005  # Fixed amount to decrease
env = gym.make("FrozenLake8x8-v1", is_slippery=False)
Q = np.zeros(
    (env.observation_space.n, env.action_space.n)
)
learned_qs = []
total_rewards = []
num_steps = []
average_total_rewards = []
average_num_steps = []
print('Q-table Initiated:')
print(Q)

for t in range(T):
    state = env.reset()
    done = False
    step = 0
    rewards = 0
    while not done:
        # print("while loop")
        action = env.action_space.sample() if np.random.random() < eps else np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)

        best_action = np.argmax(Q[next_state])
        G = reward + gamma * Q[next_state][best_action]
        GG = G - Q[state][action]
        Q[state][action] += alpha * GG

        step += 1
        rewards += reward
        state = next_state
    num_steps.append(step)
    total_rewards.append(rewards)
    if t % 50 == 0:
        average_total_rewards.append(np.array(total_rewards[-50:]).mean())
        average_num_steps.append(np.array(num_steps[-50:]).mean())
    eps = eps * (1 - eps_decay)
    learned_qs.append(Q.copy())

print('===========================================')
print('Q-table after training:')
print(Q)

fig, ax = plt.subplots(1, 1)
ax.set_xlabel("episode (*50)")
ax.set_ylabel("avg steps")
plt.plot(average_num_steps)
fig.savefig("avg_steps.png")

fig, ax = plt.subplots(1, 1)
ax.set_xlabel("episode (*50)")
ax.set_ylabel("avg total reward")
plt.plot(average_total_rewards)
fig.savefig("avg_rewards.png")

optim_Q = np.load(open("Q.npy","rb"))
RMSE = lambda q: np.sqrt(np.square(np.subtract(q[0], q[1])).mean())
err = [RMSE((optim_Q,q)) for q in learned_qs[::50]]

fig, ax = plt.subplots(1, 1)
ax.set_xlabel("episode (*50)")
ax.set_ylabel("rmse")
plt.plot(err)
fig.savefig("rmse.png")

