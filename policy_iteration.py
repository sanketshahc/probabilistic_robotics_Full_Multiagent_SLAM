import gym
import numpy as np


def evaluation(pi, env, g=1.0, theta=1e-9, max_iterations=1e6):
    V = np.zeros(env.observation_space.n)
    for i in range(int(max_iterations)):
        delta = 0
        for st in range(env.observation_space.n):
            v = 0
            for act, act_prob in enumerate(pi[st]):
                for st_prob, next_state, reward, done in env.P[st][act]:
                    v += act_prob * st_prob * (reward + g * V[next_state])

            delta = max(delta, np.abs(V[st] - v))
            V[st] = v

        if delta < theta:
            return V


def get_next(env, state, V, g):
    n = env.action_space.n
    vals = np.zeros(n)
    for action in range(n):
        for prob, next_state, r, _ in env.P[state][action]:
            vals[action] += prob * (r + g * V[next_state])
    return vals


def iteration(env, g=1.0, max_iter=5000):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    pi = np.ones([n_s, n_a]) / n_a
    evaluated_policies = 1
    for i in range(int(max_iter)):
        V = evaluation(pi, env, g=g)
        for st in range(n_s):
            action = np.argmax(pi[st])
            action_value = get_next(env, st, V, g)
            best_action = np.argmax(action_value)
            if action != best_action:
                pi[st] = np.eye(n_a)[best_action] #hot vector method
        evaluated_policies += 1
        return pi, V


def evaluate_policy(env, T, policy, render = False):
    total_reward = 0
    for _ in range(T):
        done = False
        state = env.reset()
        while not done:
            if render:
                env.render()
            action = np.argmax(policy[state])
            state, reward, done, _ = env.step(action)
            total_reward += reward
    average_reward = total_reward / T
    return total_reward, average_reward


def print_sequence():
    environment = gym.make('FrozenLake8x8-v1', is_slippery=False)
    policy, V = iteration(environment)
    actions = {
            0: "Left",
            1: "Down",
            2: "Right",
            3: "Up"
        }
    state = 0
    for _ in range(environment.observation_space.n):
        print("now in state:", state)
        action = np.argmax(policy[state])
        print("move:", actions[action])
        if not environment.P[state][action][0][-1]:
            # print(environment.P[t][action][0][-1])
            state = environment.P[state][action][0][1]
        else:
            print("final state:", state)
            print("Done:", environment.P[state][action][0][-1])
            break

T = 5000
environment = gym.make('FrozenLake8x8-v1', is_slippery=True)
policy, V = iteration(environment)
total_reward, average_reward = evaluate_policy(environment, T, policy)
print(f'Total reward = {total_reward}')
print(f'Avg reward = {average_reward}')
np.save("Q",policy)
# print_sequence()

