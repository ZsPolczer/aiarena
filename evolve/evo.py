# evo_arena/evolve/evo.py
import numpy as np
import random # For random opponent selection

from agents.brain import TinyNet
from arena.arena import Arena 

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
                        # Special case: if population is 1, it can't play against others.
                        # Fitness remains 0 or assign a baseline if desired.
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
        
        # print(f"Finished evaluating population for generation {self.generation}.") # Moved to run_evolution


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
                # Elites are directly copied. If they were mutable objects that crossover/mutation modified in-place,
                # we'd need deep copies. TinyNet's mutate/crossover return new instances, so this is okay.
                new_population.append(self.population[i]) 

        # 2. Generate the rest of the population
        num_offspring_needed = self.population_size - len(new_population) # Correctly calculate based on elites added
        
        # Parent pool: e.g., top 50% of the current (sorted) population
        # Ensure parent_pool_size is at least 1 if population exists, and not more than population size.
        parent_pool_actual_size = len(self.population)
        parent_pool_selection_limit = max(1, parent_pool_actual_size // 2)
        if parent_pool_actual_size == 0 : # Should not happen if we check self.population
            return

        for _ in range(num_offspring_needed):
            # Simple tournament-like selection from the top half (or whole pop if small)
            idx1 = self.rng.integers(0, parent_pool_selection_limit)
            idx2 = self.rng.integers(0, parent_pool_selection_limit)
            
            parent1 = self.population[idx1]
            # Ensure p2 is different if pool allows, otherwise can be same if pool is 1
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
        # print(f"Created new population for generation {self.generation + 1}.") # Moved to run_evolution

    def run_evolution(self, num_generations):
        """
        Runs the main evolutionary loop for a specified number of generations.
        """
        print(f"Starting evolution for {num_generations} generations using dt={self.match_dt:.4f}.")
        if not self.population:
            self.initialize_population()

        for gen_idx in range(num_generations):
            self.generation = gen_idx # Current generation number (0 to num_generations-1)
            print(f"\n--- Generation {self.generation}/{num_generations-1} ---")

            self.evaluate_population()
            print(f"Finished evaluating population for generation {self.generation}.")


            if not self.population: # Should not happen if initialized
                print("Error: Population became empty during evolution.")
                break

            # Sort population by fitness (descending) for logging and elitism
            self.population.sort(key=lambda genome: genome.fitness, reverse=True)
            
            if self.population: # Check if population is not empty after sort
                best_fitness = self.population[0].fitness 
                avg_fitness = sum(g.fitness for g in self.population) / len(self.population) if self.population else 0.0
                print(f"Stats: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")
            else:
                 print("Warning: Population is empty after evaluation/sorting.")


            # Save best genome of this generation (optional)
            # if self.population:
            #     persist.save_genome(self.population[0], filename_prefix=f"gen_{self.generation}_best", ...)

            if gen_idx < num_generations - 1: # Don't reproduce after the last generation
                self.select_and_reproduce()
                print(f"Created new population for generation {self.generation + 1}.")
        
        # Final evaluation and stats for the very last population
        if self.population: # Ensure there's a population to evaluate
            print(f"\n--- Final Evaluation of population from generation {num_generations-1} ---")
            self.evaluate_population() # Evaluate the final set of genomes
            self.population.sort(key=lambda genome: genome.fitness, reverse=True)
            if self.population:
                best_fitness = self.population[0].fitness
                avg_fitness = sum(g.fitness for g in self.population) / len(self.population) if self.population else 0.0
                print(f"Final Population Stats: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")

        print("\nEvolution finished.")
        return self.population 

if __name__ == '__main__':
    print("Testing EvolutionOrchestrator...")
    
    evo_orchestrator = EvolutionOrchestrator(
        population_size=10,       
        num_elites=2,
        mutation_sigma=0.3,
        num_eval_matches_per_genome=2, 
        match_max_steps=500, # Shorter matches for quick test
        match_dt=1.0/20.0,   # Test with a specific dt
        default_agent_hp=100 
    )

    # Run for a few generations
    final_population = evo_orchestrator.run_evolution(num_generations=3)

    print("\n--- Top genomes from final population: ---")
    if final_population:
        for i in range(min(3, len(final_population))): 
            genome = final_population[i]
            print(f"Rank {i+1}: Fitness = {genome.fitness:.4f}")