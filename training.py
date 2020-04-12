import logging
import time

from agent import Agent, NeuralAgent
from environment import Environment

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)


def play_episode(env: Environment, agent: Agent):
    states, actions, rewards = [], [], []

    with env:
        state, reward, score, game_over = env.reset()

        while not game_over:
            action = agent.action(state)

            states.append(state)
            actions.append(action)

            state, reward, score, game_over = env.step(action)

            rewards.append(reward)

            logging.info(f"Score: {score}, Reward: {reward}, Action: {action}")

    return states, actions, rewards


if __name__ == '__main__':
    env = Environment()
    agent = NeuralAgent(445 * 490 * 2, 20, 3)

    logging.warning("Episode start in 3 seconds, move focus to IcyTower")
    time.sleep(3)

    s, a, r = play_episode(env, agent)
    print(len(s))
    print(len(a))
    print(len(r))
