# evo_arena/main.py
import pygame
import argparse
import os # For checking file existence
from arena.arena import Arena
from agents.body import AgentBody
from agents.brain import TinyNet 
from ui.viewer import Viewer
from evolve.evo import EvolutionOrchestrator 
from storage import persist 

# --- Configuration ---
ARENA_WIDTH = 800
ARENA_HEIGHT = 800
FPS = 50 

AGENT_BASE_SPEED = 150 
AGENT_ROTATION_SPEED_DPS = 180 
AGENT_RADIUS = 15
WALL_BOUNCE_LOSS_FACTOR = 0.9

MANUAL_AGENT_COLOR = (0, 150, 255) 
DUMMY_AGENT_COLOR = (255, 100, 0)  
AI_AGENT_COLOR = (0, 200, 50)     
AI_AGENT_COLOR_2 = (150, 50, 200) # For second AI in match mode

WEAPON_RANGE = 150            
WEAPON_ARC_DEG = 90           
WEAPON_COOLDOWN_TIME = 0.6    
WEAPON_DAMAGE = 25   

DEFAULT_AGENT_HP_MAIN = 100         

# --- Default Evolution Parameters ---
DEFAULT_GENERATIONS = 20
DEFAULT_POPULATION_SIZE = 32 
DEFAULT_NUM_ELITES = 4      
DEFAULT_MUTATION_SIGMA = 0.2
DEFAULT_EVAL_MATCHES = 4    
DEFAULT_MATCH_MAX_STEPS = 3000 

def run_manual_simulation():
    """Runs the simulation with one manual agent, one dummy, and one AI agent."""
    game_arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, wall_bounce_loss_factor=WALL_BOUNCE_LOSS_FACTOR)

    manual_agent = AgentBody(
        x=100, y=ARENA_HEIGHT / 2, angle_deg=0, base_speed=AGENT_BASE_SPEED,
        rotation_speed_dps=AGENT_ROTATION_SPEED_DPS, radius=AGENT_RADIUS,
        color=MANUAL_AGENT_COLOR, agent_id="player", team_id=1, hp=DEFAULT_AGENT_HP_MAIN, brain=None, 
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE
    )
    game_arena.add_agent(manual_agent)

    dummy_target = AgentBody(
        x=ARENA_WIDTH - 100, y=ARENA_HEIGHT / 2, angle_deg=180, base_speed=0,
        rotation_speed_dps=0, radius=AGENT_RADIUS + 5, color=DUMMY_AGENT_COLOR,
        agent_id="dummy", team_id=2, hp=200, is_dummy=True, brain=None, 
        weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0
    )
    game_arena.add_agent(dummy_target)
    
    ai_brain_random = TinyNet() 
    ai_agent = AgentBody(
        x=ARENA_WIDTH / 2, y=100, angle_deg=90, base_speed=AGENT_BASE_SPEED * 0.8,
        rotation_speed_dps=AGENT_ROTATION_SPEED_DPS * 0.7, radius=AGENT_RADIUS,
        color=AI_AGENT_COLOR, agent_id="ai_random_1", team_id=3, hp=DEFAULT_AGENT_HP_MAIN, brain=ai_brain_random, 
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE
    )
    game_arena.add_agent(ai_agent)

    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT, game_arena, title="Evo Arena - Manual & AI Test")
    game_viewer.run_simulation_loop(FPS, manual_agent_id="player")


def run_training_session(generations, population_size, num_elites, mutation_sigma, eval_matches, match_steps):
    """
    Initializes and runs the evolutionary training process.
    """
    print("\n" + "="*30)
    print(" STARTING EVOLUTIONARY TRAINING ")
    # ... (rest of the function remains the same)
    print(f"Default Agent HP for Eval: {DEFAULT_AGENT_HP_MAIN}") 
    print("="*30 + "\n")

    evo_orchestrator = EvolutionOrchestrator(
        population_size=population_size,
        num_elites=num_elites,
        mutation_sigma=mutation_sigma,
        arena_width=ARENA_WIDTH, 
        arena_height=ARENA_HEIGHT,
        match_max_steps=match_steps,
        match_dt=(1.0/FPS), 
        num_eval_matches_per_genome=eval_matches,
        default_agent_hp=DEFAULT_AGENT_HP_MAIN 
    )

    final_population = evo_orchestrator.run_evolution(num_generations=generations)

    if final_population:
        final_population.sort(key=lambda genome: genome.fitness, reverse=True)
        best_overall_genome = final_population[0]
        
        print(f"\nTraining complete. Best overall fitness: {best_overall_genome.fitness:.4f}")
        
        try:
            saved_path = persist.save_genome(
                best_overall_genome, 
                filename_prefix="best_trained_genome", 
                directory="storage/genomes", 
                generation=generations, 
                fitness=best_overall_genome.fitness
            )
            print(f"Saved best overall genome to: {saved_path}")
        except Exception as e:
            print(f"Error saving best genome: {e}")
            print("Please ensure the 'storage/genomes' directory exists.")
    else:
        print("Training completed, but no final population data available.")
    print("="*30 + "\n")

def run_visual_match(genome_path1, genome_path2):
    """
    Loads two genomes and runs a visual match between them.
    """
    print(f"\nRunning visual match: {os.path.basename(genome_path1)} vs {os.path.basename(genome_path2)}")
    try:
        brain1 = persist.load_genome(genome_path1)
        brain2 = persist.load_genome(genome_path2)
    except FileNotFoundError as e:
        print(f"Error loading genome: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading genomes: {e}")
        return

    game_arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, wall_bounce_loss_factor=WALL_BOUNCE_LOSS_FACTOR)

    agent1 = AgentBody(
        x=150, y=ARENA_HEIGHT / 2, angle_deg=0, base_speed=AGENT_BASE_SPEED,
        rotation_speed_dps=AGENT_ROTATION_SPEED_DPS, radius=AGENT_RADIUS,
        color=AI_AGENT_COLOR, agent_id="ai_agent_1", team_id=1, hp=DEFAULT_AGENT_HP_MAIN, brain=brain1,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE
    )
    game_arena.add_agent(agent1)

    agent2 = AgentBody(
        x=ARENA_WIDTH - 150, y=ARENA_HEIGHT / 2, angle_deg=180, base_speed=AGENT_BASE_SPEED,
        rotation_speed_dps=AGENT_ROTATION_SPEED_DPS, radius=AGENT_RADIUS,
        color=AI_AGENT_COLOR_2, agent_id="ai_agent_2", team_id=2, hp=DEFAULT_AGENT_HP_MAIN, brain=brain2,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE
    )
    game_arena.add_agent(agent2)

    title = f"Match: {os.path.basename(genome_path1).split('.')[0]} vs {os.path.basename(genome_path2).split('.')[0]}"
    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT, game_arena, title=title)
    game_viewer.run_simulation_loop(FPS) # No manual agent ID needed

def run_show_genome(genome_path, scenario='vs_dummies'):
    """
    Loads a single genome and shows its performance in a predefined scenario.
    """
    print(f"\nShowing genome: {os.path.basename(genome_path)} in scenario: {scenario}")
    try:
        ai_brain = persist.load_genome(genome_path)
    except FileNotFoundError as e:
        print(f"Error loading genome: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading genome: {e}")
        return

    game_arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, wall_bounce_loss_factor=WALL_BOUNCE_LOSS_FACTOR)

    # AI Agent (the one being showcased)
    ai_agent_show = AgentBody(
        x=ARENA_WIDTH / 2, y=ARENA_HEIGHT - 150, angle_deg=-90, # Start at bottom, facing up
        base_speed=AGENT_BASE_SPEED, rotation_speed_dps=AGENT_ROTATION_SPEED_DPS,
        radius=AGENT_RADIUS, color=AI_AGENT_COLOR, agent_id="ai_showcased", team_id=1,
        hp=DEFAULT_AGENT_HP_MAIN, brain=ai_brain,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE
    )
    game_arena.add_agent(ai_agent_show)

    if scenario == 'vs_dummies':
        # Add one or more dummy targets
        dummy1 = AgentBody(
            x=ARENA_WIDTH / 2, y=150, angle_deg=90, base_speed=0, rotation_speed_dps=0,
            radius=AGENT_RADIUS + 5, color=DUMMY_AGENT_COLOR, agent_id="dummy_1", team_id=2,
            hp=150, is_dummy=True, brain=None,
            weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0
        )
        game_arena.add_agent(dummy1)

        dummy2 = AgentBody(
            x=ARENA_WIDTH / 4, y=ARENA_HEIGHT / 2, angle_deg=0, base_speed=0, rotation_speed_dps=0,
            radius=AGENT_RADIUS + 5, color=DUMMY_AGENT_COLOR, agent_id="dummy_2", team_id=2,
            hp=100, is_dummy=True, brain=None,
            weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0
        )
        game_arena.add_agent(dummy2)
        
        dummy3 = AgentBody(
            x=ARENA_WIDTH * 3/4, y=ARENA_HEIGHT / 2, angle_deg=180, base_speed=0, rotation_speed_dps=0,
            radius=AGENT_RADIUS + 5, color=DUMMY_AGENT_COLOR, agent_id="dummy_3", team_id=2,
            hp=100, is_dummy=True, brain=None,
            weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0
        )
        game_arena.add_agent(dummy3)

    # Add more scenarios here if needed:
    # elif scenario == 'solo_navigation':
    #     # Just the AI agent, maybe with some obstacles if arena supported them
    #     pass 

    title = f"Showcase: {os.path.basename(genome_path).split('.')[0]} ({scenario})"
    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT, game_arena, title=title)
    game_viewer.run_simulation_loop(FPS) # No manual agent ID

# --- Main execution flow ---
def main():
    parser = argparse.ArgumentParser(description="Evo Arena: A simple agent evolution project.")
    parser.add_argument('mode', nargs='?', default='manual', choices=['manual', 'train', 'show', 'match'],
                        help="Mode to run: 'manual', 'train', 'show' a genome, or 'match' two genomes.")
    
    # Arguments for 'train' mode
    parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS, 
                        help=f"Number of generations to train for (default: {DEFAULT_GENERATIONS}).")
    parser.add_argument('--pop_size', type=int, default=DEFAULT_POPULATION_SIZE,
                        help=f"Population size for training (default: {DEFAULT_POPULATION_SIZE}).")
    parser.add_argument('--elites', type=int, default=DEFAULT_NUM_ELITES,
                        help=f"Number of elite genomes to carry over (default: {DEFAULT_NUM_ELITES}).")
    parser.add_argument('--mut_sigma', type=float, default=DEFAULT_MUTATION_SIGMA,
                        help=f"Mutation sigma (std dev for noise) (default: {DEFAULT_MUTATION_SIGMA}).")
    parser.add_argument('--eval_matches', type=int, default=DEFAULT_EVAL_MATCHES,
                        help=f"Number of evaluation matches per genome (default: {DEFAULT_EVAL_MATCHES}).")
    parser.add_argument('--match_steps', type=int, default=DEFAULT_MATCH_MAX_STEPS,
                        help=f"Max steps per evaluation match (default: {DEFAULT_MATCH_MAX_STEPS}).")

    # Arguments for 'match' mode
    parser.add_argument('--g1', type=str, dest='genome1_path', help="Path to the first genome file (.npz) for 'match' mode.")
    parser.add_argument('--g2', type=str, dest='genome2_path', help="Path to the second genome file (.npz) for 'match' mode.")

    # Arguments for 'show' mode
    parser.add_argument('--genome', type=str, dest='show_genome_path', help="Path to the genome file (.npz) for 'show' mode.")
    parser.add_argument('--scenario', type=str, default='vs_dummies', choices=['vs_dummies'], # Add more choices later
                        help="Scenario for 'show' mode (default: vs_dummies).")


    args = parser.parse_args()

    if args.mode == 'manual':
        print("Running in MANUAL mode (with a test AI agent).")
        run_manual_simulation()
    elif args.mode == 'train':
        run_training_session(
            generations=args.generations,
            population_size=args.pop_size,
            num_elites=args.elites,
            mutation_sigma=args.mut_sigma,
            eval_matches=args.eval_matches,
            match_steps=args.match_steps
        )
    elif args.mode == 'match':
        if not args.genome1_path or not args.genome2_path:
            parser.error("'match' mode requires --g1 and --g2 arguments specifying genome file paths.")
        elif not os.path.exists(args.genome1_path):
            parser.error(f"Genome file not found for --g1: {args.genome1_path}")
        elif not os.path.exists(args.genome2_path):
            parser.error(f"Genome file not found for --g2: {args.genome2_path}")
        else:
            run_visual_match(args.genome1_path, args.genome2_path)
            
    elif args.mode == 'show':
        if not args.show_genome_path:
            parser.error("'show' mode requires --genome argument specifying a genome file path.")
        elif not os.path.exists(args.show_genome_path):
            parser.error(f"Genome file not found for --genome: {args.show_genome_path}")
        else:
            run_show_genome(args.show_genome_path, scenario=args.scenario)
    else:
        # This case should ideally not be reached due to 'nargs=?' and 'default'
        print(f"Mode '{args.mode}' not recognized or not fully implemented yet. Running manual simulation.")
        run_manual_simulation()

if __name__ == '__main__':
    main()