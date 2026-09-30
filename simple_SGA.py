import torch
import numpy as np
import matplotlib.pyplot as plt
from policy_network import PolicyNet, encode_state, action_map
from gradient_estimate import gradient_estimate, gamma, N
from simulator import simulator

num_iterations = 1000
learning_rate = 0.001


def _moving_average(values, window=20):
    if len(values) < window:
        return None
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def simple_SGA():
    policy_net = PolicyNet()

    iterations_log = []
    reward_log = []

    for k in range(num_iterations):
        policy_net.zero_grad()

        objective, discounted_reward = gradient_estimate(policy_net)
        objective.backward()

        with torch.no_grad():
            for param in policy_net.parameters():
                param += learning_rate * param.grad

        iterations_log.append(k)
        reward_log.append(discounted_reward)
        print(f"Iteration {k:4d} | Discounted Reward: {discounted_reward:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(iterations_log, reward_log, linewidth=2, color='steelblue', label='Discounted Reward')
    smoothed = _moving_average(reward_log, window=20)
    if smoothed is not None:
        plt.plot(
            range(19, len(reward_log)),
            smoothed,
            linewidth=2,
            color='darkorange',
            label='Moving average (20)',
        )
    plt.xlabel("Iteration")
    plt.ylabel("Discounted Reward")
    plt.title("Simple SGA - Learning Curve")
    if smoothed is not None:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sga_learning_curve.png", dpi=150)
    plt.close()
    print("Saved sga_learning_curve.png")

    return policy_net, reward_log


if __name__ == "__main__":
    print("Starting Simple SGA training...")
    simple_SGA()
    print("Simple SGA training complete!")