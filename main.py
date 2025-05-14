# evo_arena/main.py
import pygame
import argparse
import os
import sys # Added for sys.argv to detect CLI usage
import glob # For finding genome files
import itertools # For generating pairs for the tournament
import shutil # For cleaning up old tournament genomes (used in evo.py but good to have if needed here)

from arena.arena import Arena
from agents.body import AgentBody
from agents.brain import TinyNet
from ui.viewer import Viewer
from evolve.evo import EvolutionOrchestrator, BEST_GENOMES_PER_GENERATION_DIR # Import the constant
from storage import persist

# --- Configuration ---
ARENA_WIDTH = 800
ARENA_HEIGHT = 800
VISUAL_FPS = 50

SIMULATION_DT = 1.0 / 30.0
MATCH_DURATION_SECONDS = 60

AGENT_BASE_SPEED = 150
AGENT_ROTATION_SPEED_DPS = 180
AGENT_RADIUS = 15
WALL_BOUNCE_LOSS_FACTOR = 0.9

MANUAL_AGENT_COLOR = (0, 150, 255)
PLAYER_TEAM_ID = 1 # For manual player
DUMMY_AGENT_COLOR = (255, 100, 0)
DUMMY_TEAM_ID = 2    # For dummies
AI_OPPONENT_COLOR = (220, 0, 0) # More reddish for opponent
AI_OPPONENT_TEAM_ID = 3 # For AI opponent
AI_AGENT_COLOR_2 = (150, 50, 200) # For second AI in match mode


WEAPON_RANGE = 150
WEAPON_ARC_DEG = 90
WEAPON_COOLDOWN_TIME = 0.6
WEAPON_DAMAGE = 25
COOLDOWN_JITTER_FACTOR = 0.1 # Default jitter for agents

DEFAULT_AGENT_HP_MAIN = 100

DEFAULT_GENERATIONS = 20
DEFAULT_POPULATION_SIZE = 32
DEFAULT_NUM_ELITES = 4
DEFAULT_MUTATION_SIGMA = 0.2
DEFAULT_EVAL_MATCHES = 4
DEFAULT_MATCH_MAX_STEPS = int(MATCH_DURATION_SECONDS / SIMULATION_DT)

GENOME_STORAGE_DIR = "storage/genomes"

# --- Core Simulation Functions ---
def run_manual_simulation(opponent_genome_path=None):
    """
    Runs the simulation with one manual agent.
    If opponent_genome_path is provided, loads a trained AI as the opponent.
    Otherwise, a random AI or just dummies are present.
    """
    game_arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, wall_bounce_loss_factor=WALL_BOUNCE_LOSS_FACTOR)

    manual_agent = AgentBody(
        x=ARENA_WIDTH / 2, y=ARENA_HEIGHT - 100, angle_deg=-90,
        base_speed=AGENT_BASE_SPEED, rotation_speed_dps=AGENT_ROTATION_SPEED_DPS, radius=AGENT_RADIUS,
        color=MANUAL_AGENT_COLOR, agent_id="player", team_id=PLAYER_TEAM_ID,
        hp=DEFAULT_AGENT_HP_MAIN + 50, brain=None,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE,
        cooldown_jitter_factor=COOLDOWN_JITTER_FACTOR
    )
    game_arena.add_agent(manual_agent)

    ai_opponent_brain = None
    ai_opponent_id = "ai_opponent_random"
    if opponent_genome_path:
        print(f"Attempting to load opponent genome from: {opponent_genome_path}")
        try:
            ai_opponent_brain = persist.load_genome(opponent_genome_path)
            ai_opponent_id = f"ai_trained_{os.path.basename(opponent_genome_path).split('.')[0]}"
            print(f"Successfully loaded trained opponent: {ai_opponent_id}")
        except FileNotFoundError:
            print(f"Warning: Opponent genome file not found at {opponent_genome_path}. Using random AI opponent.")
            ai_opponent_brain = TinyNet()
        except Exception as e:
            print(f"Warning: Error loading opponent genome ({e}). Using random AI opponent.")
            ai_opponent_brain = TinyNet()
    else:
        print("No opponent genome specified. Using random AI opponent.")
        ai_opponent_brain = TinyNet()

    ai_opponent = AgentBody(
        x=ARENA_WIDTH / 2, y=100, angle_deg=90,
        base_speed=AGENT_BASE_SPEED * 0.9, rotation_speed_dps=AGENT_ROTATION_SPEED_DPS * 0.9,
        radius=AGENT_RADIUS, color=AI_OPPONENT_COLOR, agent_id=ai_opponent_id,
        team_id=AI_OPPONENT_TEAM_ID, hp=DEFAULT_AGENT_HP_MAIN, brain=ai_opponent_brain,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE,
        cooldown_jitter_factor=COOLDOWN_JITTER_FACTOR
    )
    game_arena.add_agent(ai_opponent)

    if not opponent_genome_path :
        dummy_target_1 = AgentBody(
            x=100, y=ARENA_HEIGHT / 2, angle_deg=0, base_speed=0,
            rotation_speed_dps=0, radius=AGENT_RADIUS, color=DUMMY_AGENT_COLOR,
            agent_id="dummy_1", team_id=DUMMY_TEAM_ID, hp=100, is_dummy=True, brain=None,
            weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0
        )
        game_arena.add_agent(dummy_target_1)
        dummy_target_2 = AgentBody(
            x=ARENA_WIDTH - 100, y=ARENA_HEIGHT / 2, angle_deg=180, base_speed=0,
            rotation_speed_dps=0, radius=AGENT_RADIUS, color=DUMMY_AGENT_COLOR,
            agent_id="dummy_2", team_id=DUMMY_TEAM_ID, hp=100, is_dummy=True, brain=None,
            weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0
        )
        game_arena.add_agent(dummy_target_2)

    title = f"Manual Play vs {ai_opponent_id}"
    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT, game_arena, title=title)
    game_viewer.run_simulation_loop(VISUAL_FPS, manual_agent_id="player")


def run_training_session(generations, population_size, num_elites, mutation_sigma, eval_matches, match_steps, sim_dt):
    print("\n" + "="*30)
    print(" STARTING EVOLUTIONARY TRAINING ")
    print("="*30)
    print(f"Parameters: Generations={generations}, Population Size={population_size}, Elites={num_elites}")
    print(f"Mutation Sigma={mutation_sigma}, Eval Matches/Genome={eval_matches}")
    print(f"Simulation DT for training: {sim_dt:.4f} ({1.0/sim_dt:.1f} ticks/sec)")
    print(f"Match Steps: {match_steps} (target duration: {match_steps * sim_dt:.1f}s)")
    print(f"Default Agent HP for Eval: {DEFAULT_AGENT_HP_MAIN}")
    print(f"Best genomes per generation will be saved to: {BEST_GENOMES_PER_GENERATION_DIR}")
    print("="*30 + "\n")

    evo_orchestrator = EvolutionOrchestrator(
        population_size=population_size,
        num_elites=num_elites,
        mutation_sigma=mutation_sigma,
        arena_width=ARENA_WIDTH,
        arena_height=ARENA_HEIGHT,
        match_max_steps=match_steps,
        match_dt=sim_dt,
        num_eval_matches_per_genome=eval_matches,
        default_agent_hp=DEFAULT_AGENT_HP_MAIN
    )

    final_population = evo_orchestrator.run_evolution(num_generations=generations)

    if final_population:
        final_population.sort(key=lambda genome: genome.fitness, reverse=True)
        best_overall_genome = final_population[0]

        print(f"\nTraining complete. Best overall fitness in final population: {best_overall_genome.fitness:.4f}")

        try:
            final_best_dir = os.path.join(GENOME_STORAGE_DIR, "final_bests")
            if not os.path.exists(final_best_dir):
                os.makedirs(final_best_dir)
                print(f"Created directory: {final_best_dir}")

            saved_path = persist.save_genome(
                best_overall_genome,
                filename_prefix="final_best_genome",
                directory=final_best_dir, # Save to a sub-directory
                generation=generations, # Total generations run
                fitness=best_overall_genome.fitness
            )
            print(f"Saved best overall genome from final population to: {saved_path}")
        except Exception as e:
            print(f"Error saving best genome from final population: {e}")
    else:
        print("Training completed, but no final population data available.")
    print("="*30 + "\n")

def run_visual_match(genome_path1, genome_path2):
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
        color=AI_OPPONENT_COLOR, agent_id="ai_agent_1", team_id=1, hp=DEFAULT_AGENT_HP_MAIN, brain=brain1,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE,
        cooldown_jitter_factor=COOLDOWN_JITTER_FACTOR
    )
    game_arena.add_agent(agent1)

    agent2 = AgentBody(
        x=ARENA_WIDTH - 150, y=ARENA_HEIGHT / 2, angle_deg=180, base_speed=AGENT_BASE_SPEED,
        rotation_speed_dps=AGENT_ROTATION_SPEED_DPS, radius=AGENT_RADIUS,
        color=AI_AGENT_COLOR_2, agent_id="ai_agent_2", team_id=2, hp=DEFAULT_AGENT_HP_MAIN, brain=brain2,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE,
        cooldown_jitter_factor=COOLDOWN_JITTER_FACTOR
    )
    game_arena.add_agent(agent2)

    title = f"Match: {os.path.basename(genome_path1).split('.')[0]} vs {os.path.basename(genome_path2).split('.')[0]}"
    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT, game_arena, title=title)
    game_viewer.run_simulation_loop(VISUAL_FPS)

def run_show_genome(genome_path, scenario='vs_dummies'):
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

    ai_agent_show = AgentBody(
        x=ARENA_WIDTH / 2, y=ARENA_HEIGHT - 150, angle_deg=-90,
        base_speed=AGENT_BASE_SPEED, rotation_speed_dps=AGENT_ROTATION_SPEED_DPS,
        radius=AGENT_RADIUS, color=AI_OPPONENT_COLOR, agent_id="ai_showcased", team_id=1,
        hp=DEFAULT_AGENT_HP_MAIN, brain=ai_brain,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE,
        cooldown_jitter_factor=COOLDOWN_JITTER_FACTOR
    )
    game_arena.add_agent(ai_agent_show)

    if scenario == 'vs_dummies':
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

    title = f"Showcase: {os.path.basename(genome_path).split('.')[0]} ({scenario})"
    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT, game_arena, title=title)
    game_viewer.run_simulation_loop(VISUAL_FPS)

# --- New Function for Post-Training Tournament ---
def run_post_training_tournament(tournament_genome_dir=None, visual=False):
    if tournament_genome_dir is None:
        # Ensure BEST_GENOMES_PER_GENERATION_DIR is defined if this is called before orchestrator init
        # It's imported from evolve.evo, so it should be available.
        tournament_genome_dir = BEST_GENOMES_PER_GENERATION_DIR

    print(f"\n--- Post-Training Tournament of Champions from '{tournament_genome_dir}' ---")

    if not os.path.exists(tournament_genome_dir):
        print(f"Tournament directory not found: {tournament_genome_dir}")
        print("Please run training first to generate per-generation best genomes, or specify a valid directory.")
        return

    genome_files = glob.glob(os.path.join(tournament_genome_dir, "*.npz"))
    if not genome_files:
        print(f"No genomes found in {tournament_genome_dir}. Cannot run tournament.")
        return

    if len(genome_files) < 2:
        print(f"Need at least 2 genomes for a tournament. Found {len(genome_files)}.")
        return

    print(f"Found {len(genome_files)} champion genomes for the tournament.")

    champions = []
    for gf_path in genome_files:
        try:
            brain = persist.load_genome(gf_path)
            # Extract generation from filename if possible for better naming
            filename_base = os.path.basename(gf_path)
            gen_part = filename_base.split('_g')[1].split('_')[0] if '_g' in filename_base else "unk"
            champ_name = f"Gen{gen_part}_{filename_base.split('_fit')[0]}" if '_fit' in filename_base else filename_base
            champions.append({'path': gf_path, 'name': champ_name, 'brain': brain, 'wins': 0, 'score': 0.0})
        except Exception as e:
            print(f"Warning: Could not load genome {gf_path}: {e}")

    if len(champions) < 2:
        print("Not enough valid champions loaded for a tournament.")
        return

    print(f"Successfully loaded {len(champions)} champions.")

    arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, wall_bounce_loss_factor=WALL_BOUNCE_LOSS_FACTOR)
    match_dt = SIMULATION_DT
    max_steps_tournament = int(MATCH_DURATION_SECONDS / match_dt)

    match_count = 0
    for champ1_idx, champ2_idx in itertools.combinations(range(len(champions)), 2):
        champ1 = champions[champ1_idx]
        champ2 = champions[champ2_idx]
        match_count +=1

        print(f"\nMatch {match_count}: {champ1['name']} vs {champ2['name']}")

        if visual:
            run_visual_match(champ1['path'], champ2['path'])
            print(f"Visual match displayed.")
            while True:
                score_input = input("Score this visual match? (Enter '1' if {} won, '2' if {} won, 'd' for draw, 's' to skip scoring for this match): ".format(champ1['name'], champ2['name'])).strip().lower()
                if score_input == '1':
                    champ1['wins'] += 1
                    champ1['score'] += 1.0
                    champ2['score'] -= 1.0
                    break
                elif score_input == '2':
                    champ2['wins'] += 1
                    champ2['score'] += 1.0
                    champ1['score'] -= 1.0
                    break
                elif score_input == 'd':
                    champ1['score'] += 0.1
                    champ2['score'] += 0.1
                    break
                elif score_input == 's':
                    break
                else:
                    print("Invalid input. Try again.")
            continue

        # Headless match for scoring
        agent_configs = [
            {'brain': champ1['brain'], 'team_id': 1, 'agent_id': champ1['name'],
             'start_pos': (150, ARENA_HEIGHT / 2, 0), 'hp': DEFAULT_AGENT_HP_MAIN},
            {'brain': champ2['brain'], 'team_id': 2, 'agent_id': champ2['name'],
             'start_pos': (ARENA_WIDTH - 150, ARENA_HEIGHT / 2, 180), 'hp': DEFAULT_AGENT_HP_MAIN}
        ]

        match_results = arena.run_match(agent_configs, max_steps_tournament, match_dt)
        winner_team_id = match_results['winner_team_id']

        if winner_team_id == 1:
            print(f"Winner: {champ1['name']}")
            champ1['wins'] += 1
            champ1['score'] += 1.0
            champ2['score'] -= 1.0
        elif winner_team_id == 2:
            print(f"Winner: {champ2['name']}")
            champ2['wins'] += 1
            champ2['score'] += 1.0
            champ1['score'] -= 1.0
        else: # Draw
            print("Result: Draw")
            champ1['score'] += 0.1
            champ2['score'] += 0.1

    champions.sort(key=lambda c: (c['score'], c['wins']), reverse=True)

    print("\n--- Tournament Results ---")
    print(f"{'Rank':<5} {'Name':<50} {'Score':<10} {'Wins':<5}")
    print("-" * 70)
    for i, champ in enumerate(champions):
        print(f"{i+1:<5} {champ['name']:<50} {champ['score']:<10.2f} {champ['wins']:<5}")

    if champions:
        print(f"\nOverall Tournament Winner: {champions[0]['name']} (Score: {champions[0]['score']:.2f}, Wins: {champions[0]['wins']})")


# --- Menu Helper Functions ---
def get_int_input(prompt, default_value):
    while True:
        try:
            val_str = input(f"{prompt} (default: {default_value}): ").strip()
            if not val_str:
                return default_value
            return int(val_str)
        except ValueError:
            print("Invalid input. Please enter a whole number.")

def get_float_input(prompt, default_value):
    while True:
        try:
            val_str = input(f"{prompt} (default: {default_value:.4f}): ").strip()
            if not val_str:
                return default_value
            return float(val_str)
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_genome_file(prompt_message, allow_none=False, none_option_text="None (random AI or no specific genome)"):
    print(f"\n{prompt_message}")

    genome_files = []
    # Check both the main genome storage and the per-generation bests for a wider selection
    search_dirs = [GENOME_STORAGE_DIR, BEST_GENOMES_PER_GENERATION_DIR, os.path.join(GENOME_STORAGE_DIR, "final_bests")]
    
    collected_paths = set() # Use a set to avoid duplicates if dirs overlap or contain same files

    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for f_name in os.listdir(s_dir):
                if f_name.endswith(".npz"):
                     full_path = os.path.join(s_dir, f_name)
                     collected_paths.add(full_path)
    
    sorted_paths = sorted(list(collected_paths), key=lambda p: os.path.basename(p))


    options = []
    if allow_none:
        options.append((None, none_option_text))

    for path in sorted_paths:
        options.append((path, f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}")) # Show parent dir and filename

    if not options and not allow_none and not sorted_paths:
         print(f"No genomes found in monitored directories. Cannot proceed without a genome.")
         while True:
            path_input = input(f"Please type a full path to a genome .npz file: ").strip()
            if os.path.exists(path_input) and path_input.endswith(".npz"):
                return path_input
            else:
                print("File not found or not a .npz file. Please try again.")

    if not options and allow_none : # Only None option left
        print(f"No genomes found in monitored directories.")
        user_path = input(f"Press Enter for '{none_option_text}', or type a full path to a genome: ").strip()
        if not user_path: return None
        if os.path.exists(user_path) and user_path.endswith(".npz"): return user_path
        print("Invalid path specified. Defaulting to 'None'.")
        return None
    elif not sorted_paths and not allow_none: # Should be caught by the above case asking for path
        return None


    for i, (path, display_name) in enumerate(options):
        print(f"{i}. {display_name}")

    while True:
        raw_choice = input(f"Select by number (0-{len(options)-1}) or type full path: ").strip()
        if os.path.exists(raw_choice) and raw_choice.endswith(".npz"):
            return raw_choice
        try:
            choice_idx = int(raw_choice)
            if 0 <= choice_idx < len(options):
                selected_path, _ = options[choice_idx]
                return selected_path
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input. Please enter a number or a valid file path.")

# --- Menu Mode Functions ---
def menu_run_manual():
    print("\n--- Manual Play Setup ---")
    opponent_genome = select_genome_file(
        "Select an AI opponent genome (optional):",
        allow_none=True,
        none_option_text="Random AI opponent / Dummies"
    )
    run_manual_simulation(opponent_genome_path=opponent_genome)

def menu_run_training():
    print("\n--- Configure Training Session ---")
    generations = get_int_input("Number of generations", DEFAULT_GENERATIONS)
    pop_size = get_int_input("Population size", DEFAULT_POPULATION_SIZE)
    elites = get_int_input("Number of elites", DEFAULT_NUM_ELITES)
    mut_sigma = get_float_input("Mutation sigma", DEFAULT_MUTATION_SIGMA)
    eval_matches = get_int_input("Evaluation matches per genome", DEFAULT_EVAL_MATCHES)
    sim_dt_chosen = get_float_input("Simulation time step (dt) for training", SIMULATION_DT)
    default_match_steps_for_chosen_dt = int(MATCH_DURATION_SECONDS / sim_dt_chosen) if sim_dt_chosen > 0 else DEFAULT_MATCH_MAX_STEPS
    match_steps = get_int_input(f"Max steps per evaluation match (for ~{MATCH_DURATION_SECONDS}s duration)", default_match_steps_for_chosen_dt)

    print(f"\nStarting training with: {generations} gens, {pop_size} pop, {elites} elites, sigma {mut_sigma:.2f}")
    print(f"Eval: {eval_matches} matches/genome. Sim DT: {sim_dt_chosen:.4f}, Match Steps: {match_steps}")

    if input("Proceed with training? (y/n): ").strip().lower() == 'y':
        run_training_session(
            generations=generations,
            population_size=pop_size,
            num_elites=elites,
            mutation_sigma=mut_sigma,
            eval_matches=eval_matches,
            match_steps=match_steps,
            sim_dt=sim_dt_chosen
        )
    else:
        print("Training cancelled.")


def menu_run_match():
    print("\n--- Visual Match Setup (AI vs AI) ---")
    print("Select the first genome (Agent 1):")
    genome1_path = select_genome_file("Choose Genome 1:", allow_none=False)
    if not genome1_path: return

    print("\nSelect the second genome (Agent 2):")
    genome2_path = select_genome_file("Choose Genome 2:", allow_none=False)
    if not genome2_path: return

    if genome1_path == genome2_path:
        print("Warning: Selected the same genome for both agents. This is allowed.")

    run_visual_match(genome1_path, genome2_path)

def menu_run_show():
    print("\n--- Show Genome Setup ---")
    genome_path = select_genome_file("Select a genome to showcase:", allow_none=False)
    if not genome_path: return
    scenario = 'vs_dummies'
    print(f"Using scenario: {scenario}")
    run_show_genome(genome_path, scenario=scenario)

def menu_run_post_tournament():
    print("\n--- Post-Training Tournament Setup ---")
    visual_tournament = input("Run tournament visually? (y/n, headless is faster for scoring): ").strip().lower() == 'y'
    if visual_tournament:
        print("Visual tournament selected. You will be prompted to score each match manually.")
    run_post_training_tournament(visual=visual_tournament)


def display_main_menu():
    print("\n===== Evo Arena Main Menu =====")
    print("1. Manual Play vs AI/Dummies")
    print("2. Train New AI Agents")
    print("3. Visual Match (AI vs AI)")
    print("4. Showcase a Trained AI Genome")
    print("5. Run Post-Training Tournament of Champions")
    print("-------------------------------")
    print("0. Exit")
    print("==============================")

def main_menu_loop():
    while True:
        display_main_menu()
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            menu_run_manual()
        elif choice == '2':
            menu_run_training()
        elif choice == '3':
            menu_run_match()
        elif choice == '4':
            menu_run_show()
        elif choice == '5':
            menu_run_post_tournament()
        elif choice == '0':
            print("Exiting Evo Arena. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

# --- Main Execution ---
def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Evo Arena: A simple agent evolution project.")
        parser.add_argument('mode', nargs='?', default=None, choices=['manual', 'train', 'show', 'match', 'tournament'],
                            help="Mode to run. If no mode, menu is shown.")

        parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS)
        parser.add_argument('--pop_size', type=int, default=DEFAULT_POPULATION_SIZE)
        parser.add_argument('--elites', type=int, default=DEFAULT_NUM_ELITES)
        parser.add_argument('--mut_sigma', type=float, default=DEFAULT_MUTATION_SIGMA)
        parser.add_argument('--eval_matches', type=int, default=DEFAULT_EVAL_MATCHES)
        parser.add_argument('--sim_dt', type=float, default=SIMULATION_DT)
        parser.add_argument('--match_steps', type=int, default=int(MATCH_DURATION_SECONDS / SIMULATION_DT))
        parser.add_argument('--opponent_genome', type=str, dest='manual_opponent_genome_path')
        parser.add_argument('--g1', type=str, dest='genome1_path')
        parser.add_argument('--g2', type=str, dest='genome2_path')
        parser.add_argument('--genome', type=str, dest='show_genome_path')
        parser.add_argument('--scenario', type=str, default='vs_dummies', choices=['vs_dummies'])
        parser.add_argument('--tournament_dir', type=str, default=BEST_GENOMES_PER_GENERATION_DIR)
        parser.add_argument('--visual_tournament', action='store_true')

        args = parser.parse_args()

        if args.mode is None:
            print("No command-line mode specified. Launching text menu...")
            main_menu_loop()
            return

        current_match_steps = args.match_steps
        if args.mode == 'train':
            sim_dt_is_custom = (args.sim_dt != SIMULATION_DT)
            # Argparse default for match_steps is based on the *code's* SIMULATION_DT constant
            match_steps_is_argparse_default_calc = (args.match_steps == int(MATCH_DURATION_SECONDS / SIMULATION_DT))
            
            if sim_dt_is_custom and match_steps_is_argparse_default_calc:
                if args.sim_dt > 0:
                    current_match_steps = int(MATCH_DURATION_SECONDS / args.sim_dt)
                    print(f"Note: --sim_dt changed. Adjusting --match_steps from {args.match_steps} to {current_match_steps} to maintain ~{MATCH_DURATION_SECONDS}s match duration.")
                else:
                    print(f"Warning: Invalid --sim_dt ({args.sim_dt}). Using default match_steps ({current_match_steps}).")
            # If user specified --match_steps, that value is used.
            # If sim_dt is default, then current_match_steps (which is args.match_steps) is also correct.


        if args.mode == 'manual':
            if args.manual_opponent_genome_path and not os.path.exists(args.manual_opponent_genome_path):
                print(f"Warning: Opponent genome for manual mode not found: {args.manual_opponent_genome_path}. Random AI used.")
                run_manual_simulation(opponent_genome_path=None)
            else:
                run_manual_simulation(opponent_genome_path=args.manual_opponent_genome_path)
        elif args.mode == 'train':
            run_training_session(
                generations=args.generations, population_size=args.pop_size, num_elites=args.elites,
                mutation_sigma=args.mut_sigma, eval_matches=args.eval_matches,
                match_steps=current_match_steps, sim_dt=args.sim_dt
            )
        elif args.mode == 'match':
            if not args.genome1_path or not args.genome2_path:
                parser.error("'match' mode requires --g1 and --g2.")
            elif not os.path.exists(args.genome1_path): parser.error(f"Genome file not found for --g1: {args.genome1_path}")
            elif not os.path.exists(args.genome2_path): parser.error(f"Genome file not found for --g2: {args.genome2_path}")
            else: run_visual_match(args.genome1_path, args.genome2_path)
        elif args.mode == 'show':
            if not args.show_genome_path: parser.error("'show' mode requires --genome.")
            elif not os.path.exists(args.show_genome_path): parser.error(f"Genome file not found for --genome: {args.show_genome_path}")
            else: run_show_genome(args.show_genome_path, scenario=args.scenario)
        elif args.mode == 'tournament':
            run_post_training_tournament(tournament_genome_dir=args.tournament_dir, visual=args.visual_tournament)
    else:
        main_menu_loop()

if __name__ == '__main__':
    main()