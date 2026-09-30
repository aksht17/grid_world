================================================================================
EE675: Introduction to Reinforcement Learning
Assignment 2 - Policy Gradient Optimization
Predator-Prey Environment (4x4 Grid)
================================================================================

OVERVIEW
--------
This folder contains the final implementation for training a predator-prey policy
using policy-gradient methods with two different optimizers: Simple Stochastic
Gradient Ascent (SGA) and Adam.

================================================================================
ENVIRONMENT DESCRIPTION
================================================================================

Grid World Setting:
  - Size: 4x4 grid
  - Initial positions: Predator at (1,1), Prey at (4,4)
  - Actions: 5 discrete actions (stay, right, left, down, up)
  - Reward: +1 when predator catches prey, 0 otherwise
  - Discount factor: gamma = 0.99
  - Episode length: Geometrically sampled (parameter 1-gamma), capped at 300 steps

Dynamics:
  - Prey respawns after being caught
  - Prey moves randomly among valid positions
  - Predator actions are sampled stochastically during training

================================================================================
FILES
================================================================================

Core Implementation:
  - policy_network.py          Neural network policy (4→64→64→5 MLP)
  - gradient_estimate.py       REINFORCE gradient estimator with full trajectory
  - simulator.py               Grid-world environment dynamics

Training Scripts:
  - simple_SGA.py              Stochastic gradient ascent optimizer (lr=0.001)
  - adam_optimizer.py          Adam optimizer (lr=0.001)

Documentation:
  - README.md                  Comprehensive documentation (Markdown format)
  - readme.txt                 This file (text format)

Output:
  - sga_learning_curve.png     Learning curve for SGA training
  - adam_learning_curve.png    Learning curve for Adam training

================================================================================
ALGORITHM: REINFORCE POLICY GRADIENT
================================================================================

Objective:
  J(θ) = E[sum_{t=0}^{T-1} log π_θ(a_t | s_t) * G_t]

where:
  - π_θ(a|s) is the policy output probability
  - G_t is the return-to-go from step t
  - T is the episode length

Key Features:
  1. Full-trajectory REINFORCE: Each action weighted by its return-to-go
  2. Geometric episode sampling: Ensures diverse trajectory lengths
  3. Per-iteration updates: One trajectory per policy update
  4. Moving average visualization: 20-step window to smooth noise

================================================================================
TRAINING CONFIGURATION
================================================================================

Both SGA and Adam use the same configuration:

  Number of iterations:         1000
  Learning rate (SGA):          0.001
  Learning rate (Adam):         0.001
  Discount factor (gamma):      0.99
  Max episode length:           300 steps
  Moving average window:        20 steps

Output:
  - Per-iteration discounted reward printed to console
  - Learning curve plot with raw rewards and moving average
  - PNG files saved (150 dpi)

================================================================================
HOW TO RUN
================================================================================

From the solution/ folder, run:

  python simple_SGA.py
  python adam_optimizer.py

Each script will:
  1. Initialize a new PolicyNet with random weights
  2. Run training for 1000 iterations
  3. Print the discounted reward for every iteration
  4. Generate and save a learning curve plot (PNG)
  5. Print completion message

Expected runtime: ~1-2 minutes per optimizer

Requirements:
  - Python 3.8 or higher
  - PyTorch 1.12 or higher
  - NumPy 1.21 or higher
  - Matplotlib 3.5 or higher

Install dependencies:
  pip install torch numpy matplotlib

================================================================================
OUTPUT FILES
================================================================================

sga_learning_curve.png
  - Blue line: Raw discounted rewards per iteration
  - Orange line: 20-step moving average
  - Shows progress of Simple SGA optimizer

adam_learning_curve.png
  - Orange line: Raw discounted rewards per iteration
  - Blue line: 20-step moving average
  - Shows progress of Adam optimizer

Both plots include:
  - Grid lines for readability
  - Legend identifying raw vs. smoothed rewards
  - X-axis: Iteration number (0-1000)
  - Y-axis: Discounted reward per trajectory
  - Title and axis labels

================================================================================
KEY IMPLEMENTATION DETAILS
================================================================================

Gradient Estimation:
  - Full trajectory REINFORCE (not single-step)
  - Each log-probability weighted by corresponding return-to-go
  - Returns normalized by empirical mean and standard deviation

Episode Length:
  - Stochastically sampled from geometric distribution
  - Parameter: 1 - gamma = 0.01
  - Ensures exploration of varying trajectory lengths
  - Capped at 300 steps

Optimization Methods:

  Simple SGA (simple_SGA.py):
    - Manual gradient ascent: param += lr * param.grad
    - Fixed learning rate (no adaptation)
    - Sensitive to gradient noise

  Adam (adam_optimizer.py):
    - PyTorch's torch.optim.Adam
    - Adaptive per-parameter learning rates
    - Better noise robustness

Visualization:
  - Raw rewards: Inherently noisy (1 trajectory per update)
  - Moving average: 20-step window filters noise
  - Reveals underlying learning trend

================================================================================
UNDERSTANDING THE OUTPUT
================================================================================

Per-iteration Output:
  Iteration    | Discounted Reward
  000          | 0.1234
  001          | 0.5678
  ...
  999          | X.XXXX

The discounted reward varies iteration-to-iteration due to:
  1. Stochastic prey movement
  2. Stochastic action sampling
  3. Single-trajectory gradient estimates

This variance is EXPECTED and NORMAL.

Learning Progress:
  - Look at the moving average line, not raw rewards
  - Adam typically shows faster convergence than SGA
  - Full convergence may require longer training (>1000 iterations)

================================================================================
NOTES AND OBSERVATIONS
================================================================================

Comparison (SGA vs Adam):
  - Adam generally achieves higher rewards due to adaptive learning rates
  - SGA is simpler but more sensitive to noise
  - Both methods implement full-trajectory REINFORCE correctly

High Variance:
  - Single-trajectory updates are inherently noisy
  - Batch REINFORCE (using multiple trajectories) would reduce noise
  - Moving average is essential for tracking improvement

Stochastic Episode Lengths:
  - Geometrically sampled episodes provide principled diversity
  - More realistic than fixed-length episodes
  - Parameter tied to discount factor

Reproducibility:
  - Results vary across runs due to random initialization and stochasticity
  - To ensure reproducibility, seed random number generators:
    
    import torch
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)

Further Improvements:
  - Batch REINFORCE: Average gradient over multiple trajectories
  - Variance reduction: Baseline subtraction or reward normalization
  - Hyperparameter tuning: Learning rate schedules, larger networks
  - Advantage estimation: Temporal difference methods (A3C, PPO, etc.)

================================================================================
TROUBLESHOOTING
================================================================================

ImportError: No module named 'torch'
  -> Install PyTorch: pip install torch

Module not found: cannot find 'simulator'
  -> Make sure you run scripts from the solution/ folder (where all files are)

Plots not saving:
  -> Check write permissions in the current directory
  -> Ensure matplotlib backend supports file I/O

Very low/zero rewards:
  -> Normal for first few iterations (random policy)
  -> Check console output for any error messages
  -> Wait for moving average to show trend

================================================================================
REPORT
================================================================================

For the full analysis, methodology, and results, see:
  assignment2_report.tex (LaTeX format)
  
The report includes:
  - Problem setup and environment rules
  - Policy network architecture
  - REINFORCE algorithm explanation
  - Implementation details for both optimizers
  - Learning curve plots with analysis
  - Discussion of SGA vs. Adam performance
  - Conclusion and insights

Compile LaTeX to PDF:
  pdflatex assignment2_report.tex

================================================================================
CONTACT & SUPPORT
================================================================================

For questions or issues, refer to:
  - solution/README.md (detailed documentation)
  - Code comments in *.py files
  - Assignment specifications
  - Course materials on REINFORCE and policy gradients

================================================================================
