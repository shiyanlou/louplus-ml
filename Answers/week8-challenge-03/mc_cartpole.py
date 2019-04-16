import gym
import numpy as np


def mc_policy_gradient(env, theta, lr, episodes):
    for episode in range(episodes):  # 迭代 episode
        episode = []
        start_observation = env.reset()  # 初始化环境
        t = 0
        while True:
            env.render()  # notebook 不支持渲染环境
            policy = np.dot(theta, start_observation)  # 计算策略值
            # 这里的 action_space 为 2, 故使用 Sigmoid 激活函数处理策略值
            pi = 1 / (1 + np.exp(-policy))
            if pi >= 0.5:
                action = 1  # 向右施加力
            else:
                action = 0  # 向左施加力
            next_observation, reward, done, _ = env.step(action)  # 执行动作
            # 将环境返回结果添加到 episode 中
            episode.append([next_observation, action, pi, reward])
            start_observation = next_observation  # 将返回 observation 作为下一次迭代 observation
            t += 1
            if done:
                print("Episode finished after {} timesteps".format(t))
                break
        # 根据上一次 episode 更新参数 theta
        for timestep in episode:
            observation, action, pi, reward = timestep
            theta += lr * (1 - pi) * np.transpose(-observation) * reward

    return theta


if __name__ == '__main__':
    lr = 0.005
    theta = np.random.rand(4)
    episodes = 10
    env = gym.make('CartPole-v1')
    mc_policy_gradient(env, theta, lr, episodes)