import torch
import numpy as np
from simulator import simulator
from policy_network import PolicyNet, encode_state, action_map

gamma = 0.99
N = 4

def gradient_estimate(policy_net, max_steps=300):
    # The geometric stopping time implicitly applies the gamma^t discount weighting
    # for infinite horizon MDPs, making the sum an unbiased estimator.
    T = np.random.geometric(1 - gamma) - 1
    T = min(T, max_steps - 1)

    pred_pos = [1, 1]
    prey_pos = [N, N]

    log_probs = []
    rewards = []

    for _ in range(T + 1):
        state_vec = encode_state(pred_pos, prey_pos, N)
        action_idx, log_prob = policy_net.select_action(state_vec)
        action = action_map[action_idx]

        next_pred, next_prey, r = simulator(N, pred_pos, prey_pos, action)
        log_probs.append(log_prob)
        rewards.append(r)

        pred_pos = next_pred
        prey_pos = next_prey

    discounted_reward = 0.0
    for t, r in enumerate(rewards):
        discounted_reward += (gamma ** t) * r

    # Monte-Carlo return-to-go G_t for each step t.
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns_t = torch.tensor(returns, dtype=torch.float32)
    objective = torch.tensor(0.0)
    for lp, G_t in zip(log_probs, returns_t):
        objective = objective + lp * G_t

    return objective, discounted_reward
