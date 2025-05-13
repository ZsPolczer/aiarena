# evo_arena/evolve/evo.py
import numpy as np
import random # For random opponent selection

from agents.brain import TinyNet
from arena.arena import Arena # To run matches for evaluation
# Import DEFAULT_AGENT_HP from arena.py if it's defined there and you prefer that route
# from arena.arena import DEFAULT_AGENT_HP as ARENA_DEFAULT_HP # Example

class EvolutionOrchestrator:
    def __init__(self,
                 population_size=64,
                 num_elites=8,
                 mutation_rate_genomes=0.9, 
                 mutation_sigma=0.2,        
                 target_fitness_stdev=0.03, 
                 
                 arena_width=800, arena_height=800,
                 match_max_steps=3000, 
                 match_dt=1.0/50.0,    
                 num_eval_matches_per_genome=4,
                 default_agent_hp=100 # <<< ADDED PARAMETER
                ):
        
        self.population_size = population_size
        self.num_elites = num_elites
        if self.num_elites > self.population_size:
            raise ValueError("Number of elites cannot exceed population size.")
            
        self.mutation_rate_genomes = mutation_rate_genomes 
        self.mutation_sigma = mutation_sigma
        self.target_fitness_stdev = target_fitness_stdev

        self.arena_width = arena_width
        self.arena_height = arena_height
        self.match_max_steps = match_max_steps
        self.match_dt = match_dt
        self.num_eval_matches_per_genome = num_eval_matches_per_genome
        self.default_agent_hp = default_agent_hp # <<< STORED PARAMETER

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
                    continue 
                
                opponent_idx = self.rng.choice(possible_opponent_indices)
                genome_B = self.population[opponent_idx]

                # Agent configs now use self.default_agent_hp when setting up agents for the match
                # if hp is not explicitly provided in the config, Arena.run_match will use its own defaults,
                # so it's important that EvolutionOrchestrator's idea of default HP for fitness normalization
                # matches what Arena.run_match uses.
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
                
                max_hp_for_norm = self.default_agent_hp # <<< CORRECTED: Use stored self.default_agent_hp

                if winner_team_id == 1: 
                    score_from_match = 1.0 
                    score_from_match += (hp_A / max_hp_for_norm) * 0.5 
                elif winner_team_id == 2: 
                    score_from_match = -1.0
                    score_from_match -= ((max_hp_for_norm - hp_B) / max_hp_for_norm) * 0.5 
                else: 
                    score_from_match = 0.0
                    hp_diff_normalized = (hp_A - hp_B) / max_hp_for_norm
                    score_from_match += hp_diff_normalized * 0.25 

                current_genome_total_score += score_from_match
            
            genome_A.fitness = current_genome_total_score / self.num_eval_matches_per_genome if self.num_eval_matches_per_genome > 0 else 0.0
        
        print(f"Finished evaluating population for generation {self.generation}.")


    def select_and_reproduce(self):
        """
        Selects parents and creates a new population through elitism, mutation, and crossover.
        """
        if not self.population:
            print("Population is empty. Cannot select and reproduce.")
            return

        self.population.sort(key=lambda genome: genome.fitness, reverse=True)
        new_population = []

        for i in range(self.num_elites):
            if i < len(self.population):
                new_population.append(self.population[i]) 

        num_offspring_needed = self.population_size - self.num_elites
        parent_pool_size = self.population_size // 2 
        if parent_pool_size == 0 and self.population_size > 0: parent_pool_size = self.population_size

        for _ in range(num_offspring_needed):
            parent1_idx = self.rng.integers(0, parent_pool_size)
            parent2_idx = self.rng.integers(0, parent_pool_size)
            while parent_pool_size > 1 and parent2_idx == parent1_idx:
                 parent2_idx = self.rng.integers(0, parent_pool_size)

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            child_genome = TinyNet.crossover(parent1, parent2, rng=self.rng)

            if self.rng.random() < self.mutation_rate_genomes:
                child_genome = child_genome.mutate(sigma=self.mutation_sigma, rng=self.rng)
            
            child_genome.fitness = 0.0 
            new_population.append(child_genome)

        self.population = new_population
        print(f"Created new population for generation {self.generation + 1}.")

    def run_evolution(self, num_generations):
        """
        Runs the main evolutionary loop for a specified number of generations.
        """
        print(f"Starting evolution for {num_generations} generations.")
        if not self.population:
            self.initialize_population()

        for gen in range(num_generations):
            self.generation = gen
            print(f"\n--- Generation {self.generation} ---")

            self.evaluate_population()

            if self.population:
                # Sort population by fitness to easily get best_fitness and for elite selection
                self.population.sort(key=lambda genome: genome.fitness, reverse=True)
                best_fitness = self.population[0].fitness 
                avg_fitness = sum(g.fitness for g in self.population) / self.population_size if self.population_size > 0 else 0.0
                print(f"Generation {self.generation} Stats: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")
                
            self.select_and_reproduce() 
        
        self.generation = num_generations # Set to final generation number for final log
        print(f"\n--- Final Evaluation (after generation {num_generations -1}) ---") # Corrected log message
        self.evaluate_population()
        if self.population:
             self.population.sort(key=lambda genome: genome.fitness, reverse=True)
             best_fitness = self.population[0].fitness
             avg_fitness = sum(g.fitness for g in self.population) / self.population_size if self.population_size > 0 else 0.0
             print(f"Final Population Stats: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")

        print("Evolution finished.")
        return self.population 

if __name__ == '__main__':
    print("Testing EvolutionOrchestrator...")
    
    evo_orchestrator = EvolutionOrchestrator(
        population_size=10,       
        num_elites=2,
        mutation_sigma=0.3,
        num_eval_matches_per_genome=2, 
        match_max_steps=500,
        default_agent_hp=100 # <<< Pass default HP for testing
    )

    final_population = evo_orchestrator.run_evolution(num_generations=3)

    print("\n--- Top genomes from final population: ---")
    if final_population:
        for i in range(min(3, len(final_population))): 
            genome = final_population[i]
            print(f"Rank {i+1}: Fitness = {genome.fitness:.4f}")