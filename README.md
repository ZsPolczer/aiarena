# Evo Arena: Agent Evolution Project

Evo Arena is a Python-based simulation environment where agents controlled by simple neural networks (TinyNets) compete and evolve. The project allows for manual control, training AI agents through evolutionary algorithms, and visualizing matches between them.

Inspired by the "Emergent রাজনিশীলতা of Multi-Agent Competition" concept, this project provides a sandbox for exploring how complex behaviors can arise from simple rules and evolutionary pressures.

## Features

*   **Simulation Arena:** A 2D arena where agents move, turn, and fire projectiles.
*   **Agent Control:**
    *   **Manual Control:** Directly control an agent using keyboard inputs.
    *   **AI Control:** Agents are controlled by a `TinyNet`, a small two-layer neural network.
*   **Neural Network (`TinyNet`):**
    *   Input: 14 normalized values representing the agent's perception of the environment (e.g., distance/bearing to nearest enemy/ally, own health, weapon status, velocity, bias).
    *   Hidden Layer: 16 neurons with `tanh` activation.
    *   Output: 4 values (thrust, strafe, rotate, fire) with `tanh` activation, thresholded to determine actions.
*   **Evolutionary Algorithm:**
    *   A genetic algorithm to evolve agent brains (`TinyNet` weights).
    *   Fitness evaluation through 1v1 matches.
    *   Selection: Elitism and selection from a parent pool.
    *   Reproduction: Uniform crossover and Gaussian mutation of network weights.
*   **Game Modes:**
    *   `manual`: Play manually against dummy/AI agents.
    *   `train`: Run the evolutionary algorithm to train AI agents.
    *   `match`: Load two saved agent brains and watch them compete visually.
    *   `show`: Load a single saved agent brain and watch it perform in a scenario (e.g., against dummies).
*   **Persistence:** Save and load evolved agent brains (`.npz` files).

## Project Structure

```
.
├── .gitignore
├── PLAN.md                 # Initial development plan
├── README.md               # This file
├── agents
│   ├── body.py             # AgentBody class (physics, actions, sensors)
│   └── brain.py            # TinyNet class (neural network)
├── arena
│   └── arena.py            # Arena class (simulation logic, rules, match runner)
├── bak                     # Backup of previous project states
├── config.py               # (Currently unused, constants are in main.py)
├── evolve
│   └── evo.py              # EvolutionOrchestrator class (evolution loop)
├── main.py                 # Main entry point, CLI argument parsing
├── storage
│   ├── genomes/            # Directory for saved agent brains (.npz files)
│   └── persist.py          # Functions for saving/loading genomes
└── ui
    └── viewer.py           # Viewer class (PyGame visualization)
```

## How It Works

### 1. Agent Body (`agents/body.py`)

*   Represents the physical agent in the arena.
*   Handles movement (thrust, strafe, rotation) based on "teleport-step" physics (velocity set each frame).
*   Manages health points (HP), weapon cooldown, and firing cone logic.
*   **Sensors (`get_inputs`):** Collects 14 specific inputs about its state and surroundings:
    1.  Forward distance to nearest enemy (normalized, +1 if none).
    2.  Sine of bearing to nearest enemy.
    3.  Cosine of bearing to nearest enemy.
    4.  Forward distance to nearest ally (normalized, +1 if none).
    5.  Sine of bearing to nearest ally.
    6.  Cosine of bearing to nearest ally.
    7.  Own health (normalized).
    8.  Weapon ready status (1 if ready, -1 if cooling).
    9.  Own x-velocity (world frame, normalized).
    10. Own y-velocity (world frame, normalized).
    11. (Unused, placeholder for future expansion, value = 0).
    12. (Unused, placeholder for future expansion, value = 0).
    13. (Unused, placeholder for future expansion, value = 0).
    14. Bias neuron (value = 1).
*   **Actuators (`perform_actions_from_outputs`):** Translates the 4 outputs from its brain into actions:
    *   Output 1 (Thrust): Forward if >= 0, backward (half speed) if < 0.
    *   Output 2 (Strafe): Left if >= 0, right if < 0 (0.75x speed).
    *   Output 3 (Rotate): Counter-clockwise if >= 0, clockwise if < 0.
    *   Output 4 (Fire): Attempt to fire if >= 0.

### 2. Agent Brain (`agents/brain.py`)

*   The `TinyNet` class is a simple feedforward neural network: 14 inputs -> 16 hidden neurons (tanh) -> 4 outputs (tanh).
*   Weights are initialized randomly or loaded from a file.
*   Includes `mutate()` (adds Gaussian noise to weights) and `crossover()` (uniform crossover) methods for evolution.

### 3. Arena (`arena/arena.py`)

*   Manages the list of agents and the simulation environment.
*   `update(dt)`:
    *   Calls each agent's `update()` method (which involves brain processing if applicable).
    *   Handles weapon firing and hit detection (checks range and firing arc).
    *   Manages wall collisions (agents bounce with a speed loss factor).
*   `run_match(agent_configs, max_duration_steps, dt)`:
    *   Sets up and runs a headless (non-visual) match between specified agents.
    *   Used by the evolution orchestrator for fitness evaluation.
    *   Returns match results (winner, duration, final agent states).
*   `check_match_end_conditions()`: Determines if a match is over (last team standing or timeout).

### 4. Evolution Orchestrator (`evolve/evo.py`)

*   `initialize_population()`: Creates an initial population of agents with random `TinyNet` brains.
*   `evaluate_population()`:
    *   For each agent (genome) in the population:
        *   It plays a set number of 1v1 matches against randomly chosen opponents from the current population.
        *   The `eval_arena.run_match()` method is used for these simulations.
        *   Fitness is calculated based on match outcomes:
            *   Win: +1.0 point + bonus for remaining HP (normalized, scaled by 0.5).
            *   Loss: -1.0 point - penalty for opponent's remaining HP (normalized, scaled by 0.5).
            *   Draw: 0.0 points + small bonus/penalty for HP difference (normalized, scaled by 0.25).
        *   The average score over these matches becomes the genome's fitness.
*   `select_and_reproduce()`:
    *   **Elitism:** The top `N` fittest genomes are copied directly to the next generation.
    *   **Parent Selection:** For the remaining spots, parents are selected from the fitter half of the population (simple random choice from this pool).
    *   **Crossover:** `TinyNet.crossover()` is used to create a child genome from two parents.
    *   **Mutation:** The child genome has a chance to be mutated using `child_genome.mutate()`.
*   `run_evolution()`: Orchestrates the cycle of evaluation, selection, and reproduction for a specified number of generations.

### 5. Viewer (`ui/viewer.py`)

*   Uses PyGame to render the arena and agents for visual modes.
*   Handles user input for manual control.
*   Displays game state information (time, match results).

### 6. Main (`main.py`)

*   Parses command-line arguments to determine the operating mode and parameters.
*   Initializes and launches the appropriate simulation or training loop.
*   Contains global constants for the simulation (arena size, agent speeds, weapon stats, default evolution parameters).

## Getting Started

### Prerequisites

*   Python (version 3.10 or higher recommended)
*   NumPy
*   PyGame

You can install the dependencies using pip:
```bash
pip install numpy pygame
```

### Running the Project

The project is run from the command line using `python main.py <mode> [options]`.

#### Modes:

1.  **`manual` (Default)**
    *   Run a visual simulation with one manually controlled agent, a dummy target, and a random AI agent.
    *   Controls: Arrow keys for movement, Spacebar to fire.
    *   Command:
        ```bash
        python main.py
        # or
        python main.py manual
        ```

2.  **`train`**
    *   Run the evolutionary algorithm to train AI agents. This is headless (no visuals during training).
    *   The best genome from the training run will be saved to `storage/genomes/`.
    *   Command:
        ```bash
        python main.py train [training_options]
        ```
    *   **Training Options:**
        *   `--generations G`: Number of generations to train (default: 20).
        *   `--pop_size P`: Population size (default: 32).
        *   `--elites E`: Number of elite genomes to carry over (default: 4).
        *   `--mut_sigma S`: Mutation sigma (standard deviation for Gaussian noise) (default: 0.2).
        *   `--eval_matches M`: Number of evaluation matches per genome per generation (default: 4).
        *   `--sim_dt DT`: Simulation time step (delta time) for headless training. Smaller values are more accurate but slower. (default: ~0.0333, i.e., 30 simulation ticks per second).
        *   `--match_steps S`: Max steps per evaluation match (default calculated for a 60-second match duration based on `sim_dt`).

        Example:
        ```bash
        python main.py train --generations 50 --pop_size 64 --sim_dt 0.02
        ```

3.  **`match`**
    *   Load two previously saved agent brains and watch them compete in a visual match.
    *   Command:
        ```bash
        python main.py match --g1 <path_to_genome1.npz> --g2 <path_to_genome2.npz>
        ```
        Example:
        ```bash
        python main.py match --g1 storage/genomes/best_trained_genome_g20_fit1.23.npz --g2 storage/genomes/another_genome.npz
        ```

4.  **`show`**
    *   Load a single saved agent brain and watch it perform in a predefined scenario (currently `vs_dummies`).
    *   Command:
        ```bash
        python main.py show --genome <path_to_genome.npz> [--scenario <scenario_name>]
        ```
        Example:
        ```bash
        python main.py show --genome storage/genomes/best_trained_genome_g20_fit1.23.npz
        # To specify a scenario (if more are added):
        # python main.py show --genome path/to/genome.npz --scenario vs_dummies
        ```

### Key Simulation Parameters (Constants in `main.py`)

*   `VISUAL_FPS`: Frames per second for visual modes (default: 50).
*   `SIMULATION_DT`: Time step for headless training simulations (default: `1.0/30.0`). A larger `dt` speeds up training but can reduce accuracy.
*   `MATCH_DURATION_SECONDS`: Target duration of simulated matches during training (default: 60).
*   `AGENT_BASE_SPEED`: Default maximum forward speed of agents (pixels/sec).
*   `AGENT_ROTATION_SPEED_DPS`: Default rotation speed (degrees/sec).
*   `WEAPON_RANGE`, `WEAPON_ARC_DEG`, `WEAPON_COOLDOWN_TIME`, `WEAPON_DAMAGE`: Default weapon characteristics.
*   `DEFAULT_AGENT_HP_MAIN`: Default health points for agents created in training and visual modes.

## Future Development / Areas for Experimentation

*   **Advanced Fitness Functions:** Incorporate more factors like damage dealt, survival time, efficiency, or specific objectives.
*   **Different Game Modes for Evolution:**
    *   Free-For-All (FFA) with more than two agents.
    *   Team-based competitions (e.g., 2v2, 3v3) with shared team fitness.
*   **More Complex Agent Sensors:**
    *   Detect walls or arena boundaries.
    *   Sense projectiles.
    *   Line-of-sight checks.
*   **More Complex Agent Actions:**
    *   Variable speed control.
    *   Special abilities or different weapon types.
*   **Obstacles or Dynamic Arenas:** Add static or moving obstacles.
*   **Advanced Evolutionary Techniques:**
    *   Different selection methods (e.g., tournament selection, roulette wheel).
    *   Speciation (e.g., NEAT-like approaches) to maintain diversity.
    *   Co-evolution: Evolve distinct roles or predator-prey dynamics.
*   **Hyperparameter Tuning:** Experiment with different population sizes, mutation rates, etc.
*   **Parallelization:** Speed up training by evaluating matches in parallel using `multiprocessing`.
*   **Improved UI/Visualization:** More detailed stats, replay saving/loading.
*   **Configuration Files:** Move simulation and evolution parameters to a configuration file (e.g., YAML) for easier management.

## Contribution

This is a hobby project. Feel free to fork, experiment, and adapt it for your own explorations into multi-agent systems and evolutionary computation.
