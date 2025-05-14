# evo_arena/evolve/evo.py
import numpy as np
import random # For random opponent selection
import os # For path operations
import shutil # For cleaning up old tournament genomes

from agents.brain import TinyNet
from arena.arena import Arena
from storage import persist # Make sure persist is imported

# Define a directory for per-generation bests
BEST_GENOMES_PER_GENERATION_DIR = "storage/genomes/per_generation_bests"

class EvolutionOrchestrator:
    def __init__(self,
                 population_size=64,
                 num_elites=8,
                 mutation_rate_genomes=0.9,
                 mutation_sigma=0.2,
                 target_fitness_stdev=0.03,

                 arena_width=800, arena_height=800,
                 match_max_steps=3000,
                 match_dt=1.0/50.0,    # Now directly takes dt
                 num_eval_matches_per_genome=4,
                 default_agent_hp=100
                ):

        self.population_size = population_size
        self.num_elites = num_elites
        if self.num_elites < 0 or self.num_elites > self.population_size: # Allow 0 elites
            raise ValueError("Number of elites must be between 0 and population size.")

        self.mutation_rate_genomes = mutation_rate_genomes
        self.mutation_sigma = mutation_sigma
        self.target_fitness_stdev = target_fitness_stdev

        self.arena_width = arena_width
        self.arena_height = arena_height
        self.match_max_steps = match_max_steps
        self.match_dt = match_dt # Store the passed dt
        self.num_eval_matches_per_genome = num_eval_matches_per_genome
        self.default_agent_hp = default_agent_hp

        self.population = []
        self.generation = 0

        self.eval_arena = Arena(self.arena_width, self.arena_height)

        self.rng = np.random.default_rng(seed=None)

        # Create the directory for per-generation bests if it doesn't exist
        if not os.path.exists(BEST_GENOMES_PER_GENERATION_DIR):
            os.makedirs(BEST_GENOMES_PER_GENERATION_DIR)
            print(f"Created directory: {BEST_GENOMES_PER_GENERATION_DIR}")
        else:
            # Optional: Clean up old per-generation bests from previous runs
            # Be careful with this if you want to keep them across multiple training sessions
            # for file_name in os.listdir(BEST_GENOMES_PER_GENERATION_DIR):
            #     file_path = os.path.join(BEST_GENOMES_PER_GENERATION_DIR, file_name)
            #     try:
            #         if os.path.isfile(file_path) or os.path.islink(file_path):
            #             os.unlink(file_path)
            #         elif os.path.isdir(file_path):
            #             shutil.rmtree(file_path)
            #     except Exception as e:
            #         print(f'Failed to delete {file_path}. Reason: {e}')
            print(f"Directory for per-generation bests already exists: {BEST_GENOMES_PER_GENERATION_DIR}")


    def initialize_population(self):
        """Creates the initial population of random TinyNet brains."""
        self.population = []
        for _ in range(self.population_size):
            brain = TinyNet()
            brain.fitness = 0.0
            self.population.append(brain)
        print(f"Initialized population with {self.population_size} random genomes.")

    def evaluate_population(self):
        """
        Evaluates the fitness of each genome in the current population.
        """
        if not self.population:
            print("Population is empty. Cannot evaluate.")
            return

        for genome in self.population:
            genome.fitness = 0.0

        for i, genome_A in enumerate(self.population):
            current_genome_total_score = 0.0

            for match_num in range(self.num_eval_matches_per_genome):
                possible_opponent_indices = [idx for idx in range(self.population_size) if idx != i]
                if not possible_opponent_indices:
                    if self.population_size == 1 and self.num_eval_matches_per_genome > 0:
                        pass
                    continue

                opponent_idx = self.rng.choice(possible_opponent_indices)
                genome_B = self.population[opponent_idx]

                agent_configs = [
                    {'brain': genome_A, 'team_id': 1, 'agent_id': f"gen{self.generation}_gA_{i}",
                     'start_pos': (100, self.arena_height / 2, 0), 'hp': self.default_agent_hp},
                    {'brain': genome_B, 'team_id': 2, 'agent_id': f"gen{self.generation}_gB_{opponent_idx}",
                     'start_pos': (self.arena_width - 100, self.arena_height / 2, 180), 'hp': self.default_agent_hp}
                ]

                match_results = self.eval_arena.run_match(agent_configs, self.match_max_steps, self.match_dt)

                score_from_match = 0.0
                winner_team_id = match_results['winner_team_id']

                hp_A = 0
                hp_B = 0
                for agent_state in match_results['agents_final_state']:
                    if agent_state['team_id'] == 1:
                        hp_A = agent_state['hp']
                    elif agent_state['team_id'] == 2:
                        hp_B = agent_state['hp']

                max_hp_for_norm = float(self.default_agent_hp) # Ensure float for division

                if winner_team_id == 1:
                    score_from_match = 1.0
                    if max_hp_for_norm > 0: score_from_match += (hp_A / max_hp_for_norm) * 0.5
                elif winner_team_id == 2:
                    score_from_match = -1.0
                    if max_hp_for_norm > 0: score_from_match -= ((max_hp_for_norm - hp_B) / max_hp_for_norm) * 0.5
                else: # Draw
                    score_from_match = 0.0
                    if max_hp_for_norm > 0:
                        hp_diff_normalized = (hp_A - hp_B) / max_hp_for_norm
                        score_from_match += hp_diff_normalized * 0.25

                current_genome_total_score += score_from_match

            genome_A.fitness = current_genome_total_score / self.num_eval_matches_per_genome if self.num_eval_matches_per_genome > 0 else 0.0


    def select_and_reproduce(self):
        """
        Selects parents and creates a new population through elitism, mutation, and crossover.
        Assumes population is already sorted by fitness (descending).
        """
        if not self.population:
            print("Population is empty. Cannot select and reproduce.")
            return

        new_population = []

        # 1. Elitism
        for i in range(self.num_elites):
            if i < len(self.population):
                new_population.append(self.population[i])

        # 2. Generate the rest of the population
        num_offspring_needed = self.population_size - len(new_population)

        parent_pool_actual_size = len(self.population)
        parent_pool_selection_limit = max(1, parent_pool_actual_size // 2)
        if parent_pool_actual_size == 0 :
            return

        for _ in range(num_offspring_needed):
            idx1 = self.rng.integers(0, parent_pool_selection_limit)
            idx2 = self.rng.integers(0, parent_pool_selection_limit)

            parent1 = self.population[idx1]
            if parent_pool_selection_limit > 1:
                while idx2 == idx1:
                    idx2 = self.rng.integers(0, parent_pool_selection_limit)
            parent2 = self.population[idx2]

            child_genome = TinyNet.crossover(parent1, parent2, rng=self.rng)

            if self.rng.random() < self.mutation_rate_genomes:
                child_genome = child_genome.mutate(sigma=self.mutation_sigma, rng=self.rng)

            child_genome.fitness = 0.0
            new_population.append(child_genome)

        self.population = new_population

    def run_evolution(self, num_generations):
        """
        Runs the main evolutionary loop for a specified number of generations.
        """
        print(f"Starting evolution for {num_generations} generations using dt={self.match_dt:.4f}.")
        if not self.population:
            self.initialize_population()

        all_time_best_fitness = -float('inf')
        all_time_best_genome_path = None

        for gen_idx in range(num_generations):
            self.generation = gen_idx # Current generation number (0 to num_generations-1)
            print(f"\n--- Generation {self.generation}/{num_generations-1} ---")

            self.evaluate_population()
            print(f"Finished evaluating population for generation {self.generation}.")


            if not self.population:
                print("Error: Population became empty during evolution.")
                break

            self.population.sort(key=lambda genome: genome.fitness, reverse=True)

            if self.population:
                current_gen_best_genome = self.population[0]
                best_fitness = current_gen_best_genome.fitness
                avg_fitness = sum(g.fitness for g in self.population) / len(self.population) if self.population else 0.0
                print(f"Stats: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")

                # --- MODIFICATION: Save best genome of this generation ---
                try:
                    saved_path_gen_best = persist.save_genome(
                        current_gen_best_genome,
                        filename_prefix="gen_best", # More specific prefix
                        directory=BEST_GENOMES_PER_GENERATION_DIR,
                        generation=self.generation, # Use current generation number
                        fitness=current_gen_best_genome.fitness
                    )
                    print(f"Saved generation {self.generation} best genome to: {saved_path_gen_best}")
                    if best_fitness > all_time_best_fitness: # Keep track of overall best path
                        all_time_best_fitness = best_fitness
                        all_time_best_genome_path = saved_path_gen_best
                except Exception as e:
                    print(f"Error saving generation {self.generation} best genome: {e}")
                # --- END MODIFICATION ---

            else:
                 print("Warning: Population is empty after evaluation/sorting.")


            if gen_idx < num_generations - 1:
                self.select_and_reproduce()
                print(f"Created new population for generation {self.generation + 1}.")

        # Final evaluation and stats for the very last population (if desired, or skip if already done)
        # Current code structure re-evaluates, which is fine.

        print("\nEvolution finished.")
        print(f"All-time best fitness during this run: {all_time_best_fitness:.4f}")
        if all_time_best_genome_path:
            print(f"Path to a genome achieving this fitness: {all_time_best_genome_path}")
        return self.population