import numpy as np

def simulator(N, predator_pos, prey_pos, action):
    dp = list(predator_pos).copy()
    pp = list(prey_pos).copy()
    r = 0

    moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    prey_move = moves[np.random.randint(0, 5)]

    ndp = [dp[0] + action[0], dp[1] + action[1]]
    if 1 <= ndp[0] <= N and 1 <= ndp[1] <= N:
        dp = ndp

    npp = [pp[0] + prey_move[0], pp[1] + prey_move[1]]
    if 1 <= npp[0] <= N and 1 <= npp[1] <= N:
        pp = npp

    if dp[0] == pp[0] and dp[1] == pp[1]:
        r = 1
        valid_respawn = []
        for x in range(1, N + 1):
            for y in range(1, N + 1):
                if x != dp[0] or y != dp[1]:
                    valid_respawn.append([x, y])
        idx = np.random.choice(len(valid_respawn))
        pp = valid_respawn[idx]

    return (dp, pp, r)
