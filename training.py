import logging
import time

from agent import Agent
from environment import Environment

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)


def play_episode(env: Environment, agent: Agent):
    with env:
        state, score, game_over = env.state()

        while not game_over:
            action = agent.action(state)
            env.act(action)
            logging.info(f"Score: {score}, Action: {action}")

            state, score, game_over = env.state()


if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    logging.warning("Episode start in 3 seconds, move focus to IcyTower")
    time.sleep(3)

    play_episode(env, agent)
