import logging
import time

import numpy as np
import torch

from agent import Agent, NeuralAgent
from environment import Environment

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)


def play_episode(env: Environment, agent: Agent, max_steps: int = 1000):
    states, actions, log_probs, scores = [], [], [], []

    with env:
        state, score, game_over = env.reset()
        step = 0

        while not game_over and step < max_steps:
            a_dir, a_jump = agent.action(state)

            states.append(state)
            actions.append((a_dir.action, a_jump.action))
            log_probs.append(torch.stack([a_dir.log_prob, a_jump.log_prob]))

            state, score, game_over = env.step(a_dir.action, a_jump.action)
            step += 1

            scores.append(score)

            logging.debug(f"Score: {score}, Action: {a_dir.action}:{a_jump.action}")

    states = np.array(states)
    actions = np.array(actions)
    log_probs = torch.stack(log_probs)
    scores = np.array(scores)
    return states, actions, log_probs, scores


def discounted_rewards(rewards, gamma=0.99):
    res = np.empty(len(rewards), dtype=np.float64)
    for i, r in enumerate(reversed(rewards)):
        prev = res[len(res) - i] if i > 0 else 0
        res[len(res) - i - 1] = r + gamma * prev

    res -= np.mean(res)
    res /= np.std(res) + 1e-7

    return torch.from_numpy(res).to(agent.device)


def learn_episode(agent: NeuralAgent, log_probs, rewards):
    agent.zero_grad()
    loss = - torch.sum(log_probs * rewards)
    loss.backward()
    agent.optimizer.step()


def get_rewards(scores):
    repair_scores(scores)

    res = scores[1:] - scores[:-1]
    res = np.concatenate([[0], res])
    return torch.from_numpy(res).to(agent.device)


def repair_scores(scores):
    for i in range(1, len(scores)):
        if scores[i] - scores[i - 1] < -30:
            scores[i] = scores[i - 1]
        if scores[i] - scores[i - 1] > 100:
            scores[i] = scores[i - 1]


if __name__ == '__main__':
    torch.set_num_threads(8)
    # torch.set_num_interop_threads(8)

    env = Environment()
    agent = NeuralAgent(445 * 490 * 2, 20, 3)

    logging.warning("Episode start in 3 seconds, move focus to IcyTower")
    time.sleep(3)

    try:
        e = 0
        while True:
            e += 1

            x, a, p, s = play_episode(env, agent, max_steps=200)
            r = get_rewards(s)
            r = discounted_rewards(r)
            learn_episode(agent, p, r)

            logging.info("e=%d steps=%d score=%d", e, len(a), np.max(s))

            del x, a, p, s, r
    except BaseException:
        with open('models/model.pkl', 'wb') as f:
            torch.save(agent, f)
        raise
