import gym

env = gym.make("CartPole-v1")
print(env.action_space)
print(env.observation_space)

print(env.observation_space.high)
print(env.observation_space.low)

for i_episode in range(20):
    # Observation reset per episode
    observation = env.reset()
    for _ in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after %s timesteps" % (_+1))
            break

env.close()