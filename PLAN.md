Okay, this is a fantastic and well-structured plan! It's ambitious but definitely achievable for a focused 1-2 day hobby project. Let's break this down into a concrete development plan, keeping it lean.

I'll follow your "Day-by-day roadmap" as the main structure and flesh out the steps within each.

---

## Development Plan

**Core Principle:** Implement the simplest thing that works for each step. Get a feature working end-to-end before over-engineering or polishing. We'll stick to NumPy and PyGame as requested.

**Global Setup (Before Day 0):**

1.  **Project Setup:**
    *   Create a main project directory (e.g., `evo_arena`).
    *   Initialize a Git repository: `git init`.
    *   Create the basic directory structure: `arena/`, `agents/`, `evolve/`, `ui/`, `storage/`.
    *   Create empty `__init__.py` files in each subdirectory to make them Python packages.
    *   Create `main.py` at the root.
    *   Create placeholder files: `arena/arena.py`, `agents/brain.py`, `agents/body.py`, `evolve/evo.py`, `ui/viewer.py`, `storage/persist.py`.
2.  **Environment:**
    *   Ensure Python ≥ 3.10 is installed.
    *   Install dependencies: `pip install numpy pygame`.
3.  **Constants:**
    *   Create a `config.py` (or similar, or just put them at the top of `arena.py` for now) to hold game constants (arena size, agent speed, weapon specs, etc.) for easy tweaking.

---

### Phase 1: Manual Control & Basic Arena (Day 0 - Evening Goal)

**Goal:** Get a PyGame window showing an arena, one manually controlled agent, and a dummy target.

1.  **`arena/arena.py` - Basic Arena Logic:**
    *   `Arena` class:
        *   `__init__(self, width=800, height=800)`: Store dimensions.
        *   `add_agent(self, agent)`: Method to add agents (initially just one manual, one dummy).
        *   `update(self, dt)`:
            *   Loop through agents and call their `update` method (to be created in `body.py`).
            *   Implement wall collision:
                *   Check if agent's new position is outside bounds.
                *   If so, "bounce back" (invert appropriate velocity component) and reduce speed by 10%.
                *   Clamp position to be within bounds after bounce to prevent sticking.
    *   *Initial `dt` will be based on PyGame's clock tick.*

2.  **`agents/body.py` - Manual Agent Body:**
    *   `AgentBody` class:
        *   `__init__(self, x, y, angle, speed, team_id=0)`: Store position, orientation (angle in radians/degrees), speed, HP (e.g., 100).
        *   `vx`, `vy`: Current velocity components.
        *   `manual_control(self, keys)`:
            *   Update `vx`, `vy` based on arrow keys (e.g., up/down for forward/backward thrust, left/right for rotation).
            *   Rotation changes `angle`.
            *   Forward/backward thrust sets `vx = cos(angle) * speed`, `vy = sin(angle) * speed` (or modifies existing velocity if you want acceleration, but spec says "teleport-step"). For now, let's make actions set desired velocity for next step.
        *   `update(self, dt)`:
            *   `self.x += self.vx * dt`
            *   `self.y += self.vy * dt`
            *   *Initially, weapon logic is not needed.*

3.  **`ui/viewer.py` - PyGame Visualization:**
    *   `Viewer` class:
        *   `__init__(self, arena)`: Store arena reference. Initialize PyGame window.
        *   `draw_arena(self)`: Draw the arena boundary.
        *   `draw_agent(self, agent)`: Draw an agent as a triangle or circle with a line indicating direction.
        *   `run_manual_loop(self, manual_agent, dummy_agent)`:
            *   Main PyGame loop:
                *   Handle events (quit, keyboard input).
                *   Pass keyboard state to `manual_agent.manual_control()`.
                *   Call `arena.update(dt)` (where `dt` is from PyGame clock, e.g., `1/50.0`).
                *   Clear screen.
                *   Call `draw_arena()`.
                *   Draw `manual_agent` and `dummy_agent`.
                *   `pygame.display.flip()`.
                *   `pygame.time.Clock().tick(50)` (for 50 Hz).

4.  **`main.py` - Entry Point for Manual Mode:**
    *   Import necessary classes.
    *   Create `Arena` instance.
    *   Create one `AgentBody` for manual control, another `AgentBody` as a static dummy target.
    *   Add agents to the arena.
    *   Create `Viewer` instance.
    *   Call `viewer.run_manual_loop()`.

---

### Phase 2: Neural Net Agents & Headless Evolution (Day 1)

**Goal:** Implement `TinyNet`, integrate it with `AgentBody`, create the evolution loop, and run it headlessly for 20 generations. View fitness.

1.  **`agents/brain.py` - Neural Network:**
    *   Copy-paste the `TinyNet` class from the spec.
    *   Initialize `rng = np.random.default_rng(seed=None)` at module level or pass it around for determinism. (For now, `np.random` is fine for speed).

2.  **`agents/body.py` - AI Agent Body Enhancements:**
    *   Modify `AgentBody`:
        *   `__init__(...)`: Add `brain` parameter (a `TinyNet` instance). Add `weapon_cooldown_timer = 0`, `max_cooldown = 0.6`.
        *   `get_inputs(self, arena)`:
            *   This is crucial. It needs to find the nearest enemy and ally (if any).
            *   For 1v1, "ally" isn't relevant yet, but design for it.
            *   Calculate distances and bearings (sin/cos θ). Normalize to \[-1, 1].
            *   Implement "±1 if none in sight" logic.
            *   Own health / 100.
            *   Weapon ready? (1 if `weapon_cooldown_timer <= 0`, -1 otherwise).
            *   x-velocity, y-velocity (normalized/clamped).
            *   Bias neuron = 1.
            *   Return the 14-element NumPy array.
        *   `act(self, outputs)`:
            *   Take the 4 outputs from `TinyNet`.
            *   Threshold them (≥ 0 vs < 0).
            *   Translate to actions:
                *   Thrust: Adjust `vx`, `vy` based on current `angle` and `speed`.
                *   Strafe: Adjust `vx`, `vy` based on `angle + pi/2` or `angle - pi/2`.
                *   Rotate: Adjust `angle`.
                *   Fire: If `weapon_cooldown_timer <= 0`, set a flag `is_firing = True` and reset `weapon_cooldown_timer = self.max_cooldown`.
        *   Modify `update(self, dt)`:
            *   If `brain` exists:
                *   `inputs = self.get_inputs(arena)`
                *   `outputs = self.brain(inputs)`
                *   `self.act(outputs)`
            *   Decrement `weapon_cooldown_timer` if > 0.
            *   `self.x += self.vx * dt`
            *   `self.y += self.vy * dt`
            *   Reset `is_firing = False` after physics step.

3.  **`arena/arena.py` - Game Logic Enhancements:**
    *   Modify `Arena`:
        *   `update(self, dt)`:
            *   After agents update their positions:
                *   Handle weapon firing:
                    *   For each agent that `is_firing`:
                        *   Check for hits on other agents (enemies).
                        *   A hit occurs if an enemy is within the 120° cone and 80m range.
                        *   (Simple collision detection: distance check + angle check).
                        *   If hit, reduce target's HP by 10.
            *   Check match end conditions:
                *   One agent (or team) left.
                *   60s timeout (`self.game_time += dt`).
                *   Return a status (e.g., winner_id, 'draw').
        *   `reset(self)`: Method to clear agents, reset game time, etc., for new matches.
        *   Need a method to run a single match: `run_match(agent1_brain, agent2_brain, max_duration=60)`:
            *   Reset arena.
            *   Create two `AgentBody` instances, assign them the provided brains.
            *   Add them to the arena.
            *   Loop `max_duration / dt` times (e.g., 60s * 50Hz = 3000 steps):
                *   `arena.update(dt)`.
                *   If match ends, break and return winner/scores.
            *   Return final state (HP of both agents, who won).

4.  **`evolve/evo.py` - Evolution Loop:**
    *   `EvolutionOrchestrator` class:
        *   `__init__(self, population_size=64, num_elites=8, mutation_rate=0.9, mutation_sigma=0.2, target_fitness_stdev=0.03)`: Store params.
        *   `population`: A list of `TinyNet` instances (genomes).
        *   `initialize_population(self)`: Create `population_size` random `TinyNet`s.
        *   `run_tournament(self, genome1, genome2, arena_instance)`:
            *   Use `arena_instance.run_match(genome1.brain, genome2.brain)` (or directly pass genome if `TinyNet` is the genome).
            *   Return fitness components for genome1.
        *   `evaluate_population(self, arena_instance)`:
            *   For each genome in `population`:
                *   Initialize its fitness to 0.
                *   Play 4 matches (1v1) against random opponents from the current population.
                *   `fitness += num_wins + 0.1 * (HP_left - HP_enemy)`.
                *   Store fitness with the genome (e.g., `genome.fitness = ...`).
        *   `select_and_reproduce(self)`:
            *   Sort population by fitness.
            *   Copy top `num_elites` to the new population.
            *   For the remaining slots:
                *   With `mutation_rate` chance:
                    *   Select one parent (e.g., from top 50%).
                    *   `new_genome = parent.mutate(sigma=self.mutation_sigma)` (add `mutate` method to `TinyNet`).
                *   Else (crossover):
                    *   Select two parents.
                    *   `new_genome = TinyNet.crossover(parent1, parent2)` (add `crossover` method to `TinyNet`).
            *   Replace old population with new.
        *   `diversity_guard(self)`:
            *   Calculate stdev of fitness scores.
            *   If stdev < `target_fitness_stdev`, increase `self.mutation_sigma` (e.g., `*= 1.1`).
        *   `evolve(self, generations=20, arena_instance)`:
            *   Main loop:
                *   `self.evaluate_population(arena_instance)`.
                *   Log best/avg fitness (print to console, or basic CSV).
                *   `self.select_and_reproduce()`.
                *   `self.diversity_guard()`.
                *   Every 10 generations: call `persist.save_genome()` for the best genome.

5.  **`agents/brain.py` - `TinyNet` modifications:**
    *   `mutate(self, sigma)`:
        *   Create new `w_in_mutated = self.w_in + np.random.normal(0, sigma, self.w_in.shape)`.
        *   Create new `w_out_mutated = self.w_out + np.random.normal(0, sigma, self.w_out.shape)`.
        *   Return `TinyNet(w_in_mutated, w_out_mutated)`.
    *   `crossover(cls, parent1, parent2)` (class method):
        *   `w_in_child = np.where(np.random.rand(*parent1.w_in.shape) < 0.5, parent1.w_in, parent2.w_in)` (uniform crossover).
        *   `w_out_child = np.where(np.random.rand(*parent1.w_out.shape) < 0.5, parent1.w_out, parent2.w_out)`.
        *   Return `TinyNet(w_in_child, w_out_child)`.

6.  **`storage/persist.py` - Genome Storage:**
    *   `save_genome(genome, filename)`:
        *   `np.savez(filename, w_in=genome.w_in, w_out=genome.w_out, fitness=getattr(genome, 'fitness', -1))`.
    *   `load_genome(filename)`:
        *   `data = np.load(filename)`.
        *   Return `TinyNet(data['w_in'], data['w_out'])`. (Fitness can be retrieved if needed).

7.  **`main.py` - `train` Command:**
    *   Add argument parsing (e.g., `argparse`) for a `train` command.
    *   `if args.command == 'train'`:
        *   Create `Arena` instance.
        *   Create `EvolutionOrchestrator` instance.
        *   `orchestrator.initialize_population()`.
        *   `orchestrator.evolve(generations=args.iters, arena_instance=arena)`.
    *   To view fitness: For now, just print to console. A separate simple script can plot the CSV later if you log to CSV.

8.  **(Stretch) `storage/persist.py` - JSONL Replay:**
    *   In `Arena.run_match()`:
        *   At each step (or every N steps), collect state: `{'t': self.game_time, 'agents': [{'id': a.id, 'x': a.x, 'y': a.y, 'angle': a.angle, 'hp': a.hp} for a in self.agents]}`.
        *   Append this as a JSON line to a replay file.
    *   **(Stretch) `ui/viewer.py` or `main.py show_replay` - Simple Playback:**
        *   Read the JSONL file.
        *   For each entry, clear PyGame screen and draw agents based on the logged state.
        *   Control playback speed.

---

### Phase 3: Interactive Matches & Advanced Modes (Day 2)

**Goal:** Visualize 1v1 matches between specific agents, add FFA and Team modes.

1.  **`main.py` - `match` Command:**
    *   Add `match <genome1.npz> <genome2.npz>` command using `argparse`.
    *   Load the two genomes using `persist.load_genome()`.
    *   Create `Arena` instance.
    *   Create `Viewer` instance.
    *   `run_visual_match(self, arena, viewer, brain1, brain2)`:
        *   Similar to `Arena.run_match()` but integrates with `Viewer` for live display.
        *   In the loop:
            *   PyGame event handling.
            *   Agents get inputs, brains decide actions, bodies `act`.
            *   `arena.update(dt)`.
            *   `viewer.draw_everything()`.
            *   `pygame.display.flip()`.
            *   `pygame.time.Clock().tick(50)`.
            *   Check for match end.
    *   Use this to pit "gen 0 vs gen 200" (by saving gen 0 best and gen 200 best).

2.  **Game Mode Flags (`--ffa`, `--teams N`) in `main.py`:**
    *   Add these flags to `argparse` for the `train` command (and potentially for a new `show` command that can run different modes).

3.  **`arena/arena.py` & `evolve/evo.py` - FFA Mode:**
    *   **Arena:**
        *   `run_match()` needs to handle N agents, not just 2.
        *   Match ends when 1 agent is left or timeout.
        *   No concept of "enemy" vs "ally" in `AgentBody.get_inputs()`'s target selection for FFA; all others are "enemies".
    *   **Evolve:**
        *   `evaluate_population()` for FFA:
            *   Instead of 1v1, could do small FFA matches (e.g., 4 agents per match).
            *   Fitness is individual (e.g., survival time, damage dealt, KOs). Simpler: fitness = 1 if win, 0 if loss, + scaled HP_left.

4.  **`arena/arena.py` & `evolve/evo.py` - Team Mode:**
    *   **AgentBody:**
        *   `__init__`: Add `team_id`.
        *   `get_inputs()`: Must correctly identify nearest enemy *and* nearest ally based on `team_id`.
    *   **Arena:**
        *   `run_match()` needs to know team assignments.
        *   Match ends when one team is eliminated or timeout.
    *   **Evolve:**
        *   `evaluate_population()` for Team mode:
            *   Setup team matches (e.g., two teams of 3).
            *   **Shared team reward:** All members of the winning team get the same fitness outcome (e.g., win/loss points, team total HP difference).
            *   Example: `run_team_match(team1_genomes, team2_genomes, arena_instance)`.

5.  **(Stretch) Polish CLI, ZIP Checkpoints, README:**
    *   **CLI:** Refine `argparse` in `main.py` for all options (iterations, headless, game modes, file paths).
    *   **ZIP Checkpoints:** Modify `persist.save_genome` to potentially save to a zip if multiple files per checkpoint (e.g., genome + metadata JSON). For now, `.npz` is fine.
    *   **README.md:** Basic instructions on how to run `train`, `match`.

---

### Final Checklist Considerations (Throughout Development):

*   **Determinism:**
    *   Use `rng = np.random.default_rng(seed)`: Instantiate one `rng` object and pass it around or make it globally accessible (e.g., in `config.py`). Use `rng.uniform()`, `rng.normal()`, `rng.choice()` etc. This is key for reproducible replays and evolution.
    *   Fixed time step (50 Hz / 20 ms) is already planned.
*   **Configuration:**
    *   CLI args are a good start. For a 1-2 day project, a YAML file might be overkill but consider it if you find yourself with too many CLI options. Key parameters like mutation rates, population size, arena size should be easily tweakable.
*   **Clarity/Docs:**
    *   Good variable names.
    *   Comments for complex logic, especially in `arena.py` (hit detection, game rules) and `evolve.py` (selection, mutation).
    *   The current spec is a great "external" doc; internal code should be readable.

---

This plan breaks down the project into manageable chunks. Focus on getting one part working before moving to the next. For example, for Day 1, get *one* agent controlled by a *random* `TinyNet` moving in the arena *before* implementing the full evolution loop. Then add a second agent for 1v1, then the evolution.

Good luck, this looks like a super fun project!