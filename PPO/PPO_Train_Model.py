import gym
from gym.envs.box2d import CarRacing

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

if __name__=='__main__':
    # Create environment
    env = lambda :  CarRacing()
    env = DummyVecEnv([env])

    # Load the trained model
    model = PPO.load('best_model.zip', env=env)
    model.set_env(env)

    # Play an episode
    episode_length = 900
    obs = env.reset()
    for i in range(episode_length):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
