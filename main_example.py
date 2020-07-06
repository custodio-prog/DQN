import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
env.close()

Experience = namedtuple(
    'Experience',
    ('state','action','next_state','reward')
)
