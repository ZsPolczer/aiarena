# evo_arena/main.py
import pygame
import argparse
import os
import sys
import glob
import itertools
import shutil
import numpy as np # For GD/RL
import random # For GD/RL data generation shuffle

from arena.arena import Arena
from agents.body import AgentBody
from agents.brain import TinyNet # Crucial for GD/RL fine-tuning
from ui.viewer import Viewer
from evolve.evo import EvolutionOrchestrator, BEST_GENOMES_PER_GENERATION_DIR
from storage import persist

# --- Configuration ---
ARENA_WIDTH = 800
ARENA_HEIGHT = 800
VISUAL_FPS = 50

SIMULATION_DT = 1.0 / 30.0 # Default for training and can be used for visual match dt
MATCH_DURATION_SECONDS = 60

AGENT_BASE_SPEED = 150
AGENT_ROTATION_SPEED_DPS = 180
AGENT_RADIUS = 15
WALL_BOUNCE_LOSS_FACTOR = 0.9

MANUAL_AGENT_COLOR = (0, 150, 255)
PLAYER_TEAM_ID = 1
DUMMY_AGENT_COLOR = (255, 100, 0)
DUMMY_TEAM_ID = 2
AI_OPPONENT_COLOR = (220, 0, 0) # For AI Agent 1 (Red)
AI_OPPONENT_TEAM_ID = 3 # Generic opponent team ID
AI_AGENT_COLOR_2 = (0, 200, 50) # For AI Agent 2 (Green)

WEAPON_RANGE = 150
WEAPON_ARC_DEG = 90
WEAPON_COOLDOWN_TIME = 0.6
WEAPON_DAMAGE = 25
COOLDOWN_JITTER_FACTOR = 0.1

DEFAULT_AGENT_HP_MAIN = 100

DEFAULT_GENERATIONS = 20
DEFAULT_POPULATION_SIZE = 32
DEFAULT_NUM_ELITES = 4
DEFAULT_MUTATION_SIGMA = 0.2
DEFAULT_EVAL_MATCHES = 4
DEFAULT_MATCH_MAX_STEPS = int(MATCH_DURATION_SECONDS / SIMULATION_DT) # For training

GENOME_STORAGE_DIR = "storage/genomes"
FINETUNED_GENOME_DIR = os.path.join(GENOME_STORAGE_DIR, "finetuned")


# --- Core Simulation Functions ---
def run_manual_simulation(opponent_genome_path=None):
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
    print("\n" + "="*30); print(" STARTING EVOLUTIONARY TRAINING "); print("="*30)
    print(f"Generations={generations}, Population={population_size}, Elites={num_elites}")
    print(f"Mutation Sigma={mutation_sigma}, Eval Matches={eval_matches}")
    print(f"Sim DT: {sim_dt:.4f} ({1.0/sim_dt:.1f} tps), Match Steps: {match_steps} (~{match_steps*sim_dt:.1f}s)")
    print(f"Default Agent HP for Eval: {DEFAULT_AGENT_HP_MAIN}")
    print(f"Best genomes saved to: {BEST_GENOMES_PER_GENERATION_DIR}")
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
        final_population.sort(key=lambda g: g.fitness, reverse=True)
        best_overall = final_population[0]
        print(f"\nTraining complete. Best overall fitness in final population: {best_overall.fitness:.4f}")
        final_best_dir = os.path.join(GENOME_STORAGE_DIR, "final_bests")
        if not os.path.exists(final_best_dir): os.makedirs(final_best_dir)
        saved_path = persist.save_genome(best_overall, "final_best_genome", final_best_dir, generations, best_overall.fitness)
        print(f"Saved best overall genome from final population to: {saved_path}")
    else:
        print("Training completed, but no final population data available.")
    print("="*30 + "\n")

# MODIFIED for Best of X games
def run_visual_match(genome_path1, genome_path2, num_games=1):
    ai1_name = os.path.basename(genome_path1).split('.')[0]
    ai2_name = os.path.basename(genome_path2).split('.')[0]
    
    print(f"\nRunning Visual Match Series (Best of {num_games}):")
    print(f"  AI 1 (Red): {ai1_name}")
    print(f"  AI 2 (Green): {ai2_name}")

    try:
        brain1 = persist.load_genome(genome_path1)
        brain2 = persist.load_genome(genome_path2)
    except Exception as e:
        print(f"Error loading genome(s): {e}"); return

    score_ai1 = 0
    score_ai2 = 0

    # Create arena once
    game_arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, wall_bounce_loss_factor=WALL_BOUNCE_LOSS_FACTOR)
    
    # Create agent bodies once, their brains and states will be managed per game
    # Team IDs 1 and 2 are used for the two AIs in a 1v1 match
    agent1_body = AgentBody(
        x=150, y=ARENA_HEIGHT / 2, angle_deg=0, base_speed=AGENT_BASE_SPEED,
        rotation_speed_dps=AGENT_ROTATION_SPEED_DPS, radius=AGENT_RADIUS,
        color=AI_OPPONENT_COLOR, agent_id=ai1_name, team_id=1, hp=DEFAULT_AGENT_HP_MAIN, brain=brain1,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE,
        cooldown_jitter_factor=COOLDOWN_JITTER_FACTOR
    )
    agent2_body = AgentBody(
        x=ARENA_WIDTH - 150, y=ARENA_HEIGHT / 2, angle_deg=180, base_speed=AGENT_BASE_SPEED,
        rotation_speed_dps=AGENT_ROTATION_SPEED_DPS, radius=AGENT_RADIUS,
        color=AI_AGENT_COLOR_2, agent_id=ai2_name, team_id=2, hp=DEFAULT_AGENT_HP_MAIN, brain=brain2,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE,
        cooldown_jitter_factor=COOLDOWN_JITTER_FACTOR
    )
    # Add agents to the arena. Their initial state is stored by add_agent.
    game_arena.add_agent(agent1_body)
    game_arena.add_agent(agent2_body)

    # Viewer is created once
    initial_viewer_title = f"Game 1/{num_games} | {ai1_name}: 0 - {ai2_name}: 0"
    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT, game_arena, title=initial_viewer_title)


    for game_num in range(1, num_games + 1):
        print(f"\n--- Starting Game {game_num} of {num_games} ---")
        
        # Reset arena (clears game time) and agents (HP, position, cooldowns etc.)
        game_arena.reset_arena_and_agents()
        # Ensure brains are still assigned (they should be, as they are part of agent_body object)
        agent1_body.brain = brain1
        agent2_body.brain = brain2
        # Explicitly set standard start positions for each game for consistency
        agent1_body.reset_state(x=150, y=ARENA_HEIGHT / 2, angle_deg=0, hp=DEFAULT_AGENT_HP_MAIN)
        agent2_body.reset_state(x=ARENA_WIDTH - 150, y=ARENA_HEIGHT / 2, angle_deg=180, hp=DEFAULT_AGENT_HP_MAIN)

        current_game_title = f"Game {game_num}/{num_games} | {ai1_name} (R): {score_ai1} - {ai2_name} (G): {score_ai2}"
        pygame.display.set_caption(current_game_title) # Update PyGame window title

        running_this_game = True
        game_over_this_game = False
        winner_message_this_game = "Game in progress..."
        
        # Use VISUAL_FPS for dt in visual matches
        match_visual_dt = 1.0 / VISUAL_FPS 
        game_max_steps = int(MATCH_DURATION_SECONDS / match_visual_dt)

        for step in range(game_max_steps):
            if not pygame.display.get_init(): # Check if user closed the window
                print("Match series aborted (window closed).")
                return

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_this_game = False; game_over_this_game = True; winner_message_this_game = "Series Aborted (Quit)"
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: # Allow Esc to skip current game in series
                        running_this_game = False; game_over_this_game = True; winner_message_this_game = "Game Skipped by User"
                        break
            if not running_this_game:
                break

            game_arena.update(match_visual_dt)

            game_viewer.screen.fill((30, 30, 30))
            game_arena.draw_bounds(game_viewer.screen)
            for agent_to_draw in game_arena.agents: # Draw all agents in the arena
                agent_to_draw.draw(game_viewer.screen)
                if agent_to_draw.is_firing_command and agent_to_draw.is_alive():
                    game_viewer.draw_firing_cone(game_viewer.screen, agent_to_draw)

            is_over, winner_team_id, message = game_arena.check_match_end_conditions(max_duration_seconds=MATCH_DURATION_SECONDS)
            if is_over:
                game_over_this_game = True
                winner_message_this_game = message
                if winner_team_id == 1: # AI1 (team_id 1) won
                    score_ai1 += 1
                    winner_message_this_game += f" Winner: {ai1_name} (Red)"
                elif winner_team_id == 2: # AI2 (team_id 2) won
                    score_ai2 += 1
                    winner_message_this_game += f" Winner: {ai2_name} (Green)"
                else: # Draw
                    winner_message_this_game += " (Draw)"
                break # End this game's simulation loop
            
            # Display current game time on screen
            time_text_surf = game_viewer.info_font.render(f"Time: {game_arena.game_time:.1f}s", True, (220, 220, 220))
            game_viewer.screen.blit(time_text_surf, (10, 10))
            
            # Display agent names and current HP
            hp_text_ai1 = f"{ai1_name} (Red) HP: {agent1_body.hp:.0f}"
            hp_text_ai2 = f"{ai2_name} (Green) HP: {agent2_body.hp:.0f}"
            ai1_hp_surf = game_viewer.info_font.render(hp_text_ai1, True, AI_OPPONENT_COLOR)
            ai2_hp_surf = game_viewer.info_font.render(hp_text_ai2, True, AI_AGENT_COLOR_2)
            game_viewer.screen.blit(ai1_hp_surf, (10, ARENA_HEIGHT - 50))
            game_viewer.screen.blit(ai2_hp_surf, (10, ARENA_HEIGHT - 30))


            pygame.display.flip()
            game_viewer.clock.tick(VISUAL_FPS)
        # End of single game simulation loop

        print(f"Game {game_num} Result: {winner_message_this_game}")
        current_series_score_str = f"{ai1_name} (Red): {score_ai1}  -  {ai2_name} (Green): {score_ai2}"
        print(f"Current Series Score: {current_series_score_str}")
        pygame.display.set_caption(f"Game {game_num} Over! {current_series_score_str} | SPACE for Next")


        if not pygame.display.get_init() or winner_message_this_game == "Series Aborted (Quit)":
            if pygame.display.get_init(): pygame.quit()
            return

        if game_num < num_games:
            waiting_for_next = True
            while waiting_for_next and pygame.display.get_init():
                # Display game over message and prompt for next game
                game_viewer.screen.fill((30,30,30)) # Clear screen for message
                msg_surf = game_viewer.font.render(winner_message_this_game, True, (255, 255, 0))
                msg_rect = msg_surf.get_rect(center=(ARENA_WIDTH / 2, ARENA_HEIGHT / 2 - 20))
                game_viewer.screen.blit(msg_surf, msg_rect)
                
                score_disp_surf = game_viewer.font.render(current_series_score_str, True, (200, 200, 200))
                score_disp_rect = score_disp_surf.get_rect(center=(ARENA_WIDTH / 2, ARENA_HEIGHT / 2 + 20))
                game_viewer.screen.blit(score_disp_surf, score_disp_rect)

                next_prompt_surf = game_viewer.info_font.render("Press SPACE for Next Game or ESC to End Series", True, (200,200,200))
                next_prompt_rect = next_prompt_surf.get_rect(center=(ARENA_WIDTH/2, ARENA_HEIGHT/2 + 60))
                game_viewer.screen.blit(next_prompt_surf, next_prompt_rect)
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_next = False; pygame.quit(); return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            waiting_for_next = False
                        if event.key == pygame.K_ESCAPE:
                            print("Ending series early by user request."); pygame.quit(); return
                game_viewer.clock.tick(15) # Lower FPS for waiting screen
        if not pygame.display.get_init(): return # If window closed while waiting

    # After all games in the series are played
    print(f"\n--- Match Series Finished ---")
    final_score_str = f"FINAL SCORE: {ai1_name} (Red) {score_ai1}  -  {score_ai2} {ai2_name} (Green)"
    print(final_score_str)
    overall_winner_str = ""
    if score_ai1 > score_ai2: overall_winner_str = f"Overall Winner: {ai1_name} (Red)"
    elif score_ai2 > score_ai1: overall_winner_str = f"Overall Winner: {ai2_name} (Green)"
    else: overall_winner_str = "Overall Series is a Draw!"
    print(overall_winner_str)
    
    if pygame.display.get_init():
        final_display_wait = True
        final_caption = f"{final_score_str} | {overall_winner_str} | Press ESC"
        pygame.display.set_caption(final_caption)
        while final_display_wait and pygame.display.get_init():
            game_viewer.screen.fill((30,30,30))
            final_score_surf = game_viewer.font.render(final_score_str, True, (255,255,0))
            final_score_rect = final_score_surf.get_rect(center=(ARENA_WIDTH/2, ARENA_HEIGHT/2 - 20))
            game_viewer.screen.blit(final_score_surf, final_score_rect)
            
            overall_winner_surf = game_viewer.font.render(overall_winner_str, True, (200,200,200))
            overall_winner_rect = overall_winner_surf.get_rect(center=(ARENA_WIDTH/2, ARENA_HEIGHT/2 + 20))
            game_viewer.screen.blit(overall_winner_surf, overall_winner_rect)

            esc_prompt_surf = game_viewer.info_font.render("Press ESC to return to menu.", True, (180,180,180))
            esc_prompt_rect = esc_prompt_surf.get_rect(center=(ARENA_WIDTH/2, ARENA_HEIGHT/2 + 60))
            game_viewer.screen.blit(esc_prompt_surf, esc_prompt_rect)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    final_display_wait = False
            game_viewer.clock.tick(15)
        pygame.quit()


def run_show_genome(genome_path, scenario='vs_dummies'):
    print(f"\nShowing genome: {os.path.basename(genome_path)} in scenario: {scenario}")
    try: ai_brain = persist.load_genome(genome_path)
    except Exception as e: print(f"Error loading genome: {e}"); return

    game_arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, wall_bounce_loss_factor=WALL_BOUNCE_LOSS_FACTOR)
    ai_agent_show = AgentBody(
        x=ARENA_WIDTH / 2, y=ARENA_HEIGHT - 150, angle_deg=-90,
        base_speed=AGENT_BASE_SPEED, rotation_speed_dps=AGENT_ROTATION_SPEED_DPS,
        radius=AGENT_RADIUS, color=AI_OPPONENT_COLOR, agent_id=os.path.basename(genome_path).split('.')[0], team_id=1,
        hp=DEFAULT_AGENT_HP_MAIN, brain=ai_brain,
        weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE,
        cooldown_jitter_factor=COOLDOWN_JITTER_FACTOR
    )
    game_arena.add_agent(ai_agent_show)
    if scenario == 'vs_dummies':
        game_arena.add_agent(AgentBody(x=ARENA_WIDTH/2, y=150, angle_deg=90, agent_id="dummy1", team_id=2, hp=150, is_dummy=True, radius=AGENT_RADIUS+5, color=DUMMY_AGENT_COLOR, base_speed=0, rotation_speed_dps=0, weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0))
        game_arena.add_agent(AgentBody(x=ARENA_WIDTH/4, y=ARENA_HEIGHT/2, angle_deg=0, agent_id="dummy2", team_id=2, hp=100, is_dummy=True, radius=AGENT_RADIUS+5, color=DUMMY_AGENT_COLOR, base_speed=0, rotation_speed_dps=0, weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0))
    title = f"Showcase: {os.path.basename(genome_path).split('.')[0]} ({scenario})"
    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT, game_arena, title=title)
    game_viewer.run_simulation_loop(VISUAL_FPS)


def run_post_training_tournament(tournament_genome_dir=None, visual=False):
    if tournament_genome_dir is None: tournament_genome_dir = BEST_GENOMES_PER_GENERATION_DIR
    print(f"\n--- Post-Training Tournament of Champions from '{tournament_genome_dir}' ---")
    if not os.path.exists(tournament_genome_dir):
        print(f"Dir not found: {tournament_genome_dir}. Run training first or specify dir."); return
    genome_files = glob.glob(os.path.join(tournament_genome_dir, "*.npz"))
    if len(genome_files) < 2: print(f"Need at least 2 genomes. Found {len(genome_files)}."); return
    print(f"Found {len(genome_files)} champion genomes.")
    champions = []
    for gf_path in genome_files:
        try:
            brain = persist.load_genome(gf_path)
            base = os.path.basename(gf_path); gen_part = base.split('_g')[1].split('_')[0] if '_g' in base else "unk"
            champ_name = f"Gen{gen_part}_{base.split('_fit')[0]}" if '_fit' in base else base.split('.')[0]
            champions.append({'path': gf_path, 'name': champ_name, 'brain': brain, 'wins': 0, 'score': 0.0})
        except Exception as e: print(f"Warning: Could not load {gf_path}: {e}")
    if len(champions) < 2: print("Not enough valid champions loaded."); return
    print(f"Loaded {len(champions)} champions.")
    arena_obj = Arena(ARENA_WIDTH, ARENA_HEIGHT, wall_bounce_loss_factor=WALL_BOUNCE_LOSS_FACTOR)
    match_dt_tourney = SIMULATION_DT
    max_steps_tourney = int(MATCH_DURATION_SECONDS / match_dt_tourney)
    for count, (c1_idx, c2_idx) in enumerate(itertools.combinations(range(len(champions)), 2)):
        c1, c2 = champions[c1_idx], champions[c2_idx]
        print(f"\nMatch {count+1}: {c1['name']} vs {c2['name']}")
        if visual:
            run_visual_match(c1['path'], c2['path'], num_games=1) # Tournament visual is 1 game
            print(f"Visual match displayed. Score manually if needed for tournament context, or rely on console output.")
            # Manual scoring for tournament context if visual flag is used this way
            while True:
                score_input = input(f"Score: '1' if {c1['name']} won, '2' if {c2['name']} won, 'd' for draw, 's' to skip scoring: ").strip().lower()
                if score_input == '1': c1['wins']+=1; c1['score']+=1.0; c2['score']-=1.0; break
                elif score_input == '2': c2['wins']+=1; c2['score']+=1.0; c1['score']-=1.0; break
                elif score_input == 'd': c1['score']+=0.1; c2['score']+=0.1; break
                elif score_input == 's': break
                else: print("Invalid input.")
            continue 
        configs = [{'brain': c1['brain'], 'team_id': 1, 'agent_id': c1['name'], 'start_pos': (150, ARENA_HEIGHT/2,0), 'hp':DEFAULT_AGENT_HP_MAIN},
                   {'brain': c2['brain'], 'team_id': 2, 'agent_id': c2['name'], 'start_pos': (ARENA_WIDTH-150, ARENA_HEIGHT/2,180), 'hp':DEFAULT_AGENT_HP_MAIN}]
        results = arena_obj.run_match(configs, max_steps_tourney, match_dt_tourney)
        winner = results['winner_team_id']
        if winner == 1: print(f"Winner: {c1['name']}"); c1['wins']+=1; c1['score']+=1.0; c2['score']-=1.0
        elif winner == 2: print(f"Winner: {c2['name']}"); c2['wins']+=1; c2['score']+=1.0; c1['score']-=1.0
        else: print("Draw"); c1['score']+=0.1; c2['score']+=0.1
    champions.sort(key=lambda c: (c['score'], c['wins']), reverse=True)
    print("\n--- Tournament Results ---")
    print(f"{'Rank':<5} {'Name':<50} {'Score':<10} {'Wins':<5}")
    print("-" * 70)
    for i, champ in enumerate(champions):
        print(f"{i+1:<5} {champ['name']:<50} {champ['score']:<10.2f} {champ['wins']:<5}")
    if champions: print(f"\nOverall Tournament Winner: {champions[0]['name']} (Score: {champions[0]['score']:.2f}, Wins: {champions[0]['wins']})")


# --- RL Fine-Tuning Functions ---
def run_self_play_episode(genome, arena_width, arena_height, match_max_steps, match_dt, default_hp):
    episode_data_for_gradients = [] 
    self_play_arena = Arena(arena_width, arena_height)
    brain_copy_for_opponent = TinyNet(w_in=genome.w_in.copy(), w_out=genome.w_out.copy(),
                                      input_size=genome.input_size, hidden_size=genome.hidden_size,
                                      output_size=genome.output_size)
    agent_configs = [
        {'brain': genome, 'team_id': 1, 'agent_id': 'player1_train',
         'start_pos': (150, arena_height / 2, 0), 'hp': default_hp},
        {'brain': brain_copy_for_opponent, 'team_id': 2, 'agent_id': 'player2_opponent',
         'start_pos': (arena_width - 150, arena_height / 2, 180), 'hp': default_hp}
    ]
    self_play_arena.agents = []
    player1_agent_body = None
    for i, config in enumerate(agent_configs):
        start_x, start_y, start_angle = config.get('start_pos')
        agent = AgentBody(
            x=start_x, y=start_y, angle_deg=start_angle, base_speed=AGENT_BASE_SPEED, 
            rotation_speed_dps=AGENT_ROTATION_SPEED_DPS, radius=AGENT_RADIUS,
            agent_id=config['agent_id'], team_id=config['team_id'], hp=config['hp'], brain=config['brain'],
            weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
            weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE,
            cooldown_jitter_factor=COOLDOWN_JITTER_FACTOR
        )
        self_play_arena.add_agent(agent)
        if config['agent_id'] == 'player1_train':
            player1_agent_body = agent

    for step in range(match_max_steps):
        if not player1_agent_body or not player1_agent_body.is_alive(): break
        current_all_agents_state = list(self_play_arena.agents)
        inputs_p1 = player1_agent_body.get_inputs(arena_width, arena_height, current_all_agents_state)
        # Ensure correct tuple unpacking from forward_pass_for_gd
        x_p1, _h_pre_p1, h_p1_activated, _y_pre_p1, y_p1_actions = genome.forward_pass_for_gd(inputs_p1)
        episode_data_for_gradients.append({'x_input': x_p1, 'h_activated': h_p1_activated, 'y_activated_actions': y_p1_actions})
        self_play_arena.update(match_dt)
        match_over, _, _ = self_play_arena.check_match_end_conditions(max_duration_seconds=(match_max_steps * match_dt))
        if match_over: break
    
    final_results = self_play_arena.check_match_end_conditions(max_duration_seconds=(match_max_steps * match_dt))
    winner_team_id = final_results[1]
    match_reward = 0.0
    if winner_team_id == 1: match_reward = 1.0
    elif winner_team_id == 2: match_reward = -1.0
        
    return episode_data_for_gradients, match_reward


def fine_tune_genome_with_rl(genome_to_tune, num_episodes=50, learning_rate=0.001,
                             arena_width=ARENA_WIDTH, arena_height=ARENA_HEIGHT,
                             match_max_steps=DEFAULT_MATCH_MAX_STEPS, match_dt=SIMULATION_DT,
                             default_hp=DEFAULT_AGENT_HP_MAIN):
    print(f"[DEBUG] fine_tune_genome_with_rl started with num_episodes: {num_episodes}, learning_rate: {learning_rate}")
    print(f"\n--- Starting RL Fine-Tuning (Self-Play) ---")
    print(f"Genome initial fitness: {genome_to_tune.fitness:.4f}")
    # ... (rest of the function as before, ensuring num_episodes and learning_rate are used)
    avg_rewards_history = []
    for episode_num in range(num_episodes):
        try:
            trajectory_data, final_match_reward = run_self_play_episode(
                genome_to_tune, arena_width, arena_height, match_max_steps, match_dt, default_hp
            )
            avg_rewards_history.append(final_match_reward)

            if not trajectory_data:
                print(f"Ep {episode_num+1}/{num_episodes}: No data. Reward: {final_match_reward:.1f}. Skip update.")
                continue
            
            total_steps_in_episode = len(trajectory_data)
            
            for step_data in trajectory_data:
                dW_in, dW_out = genome_to_tune.get_policy_gradient_for_action(
                    step_data['x_input'], 
                    step_data['h_activated'], 
                    step_data['y_activated_actions'], 
                    final_match_reward 
                )
                effective_lr = learning_rate / total_steps_in_episode if total_steps_in_episode > 1 else learning_rate
                genome_to_tune.update_weights(dW_in, dW_out, effective_lr)

            if (episode_num + 1) % (max(1, num_episodes // 10)) == 0 or episode_num == 0:
                outcome_str = "WIN" if final_match_reward > 0 else "LOSS" if final_match_reward < 0 else "DRAW"
                avg_rew = np.mean(avg_rewards_history[-(max(1,num_episodes // 20)):]) if avg_rewards_history else 0.0
                print(f"Ep {episode_num+1}/{num_episodes}: Outcome: {outcome_str} ({final_match_reward:.1f}), Steps: {total_steps_in_episode}, AvgRew (last N): {avg_rew:.2f}")
        except Exception as e:
            print(f"Error in RL tuning episode {episode_num+1}: {e}")
            import traceback
            traceback.print_exc()
            break 
    print("--- RL Fine-Tuning Finished ---")
    return genome_to_tune


# --- Menu Helper Functions ---
def get_int_input(prompt, default_value):
    while True:
        try: val_str = input(f"{prompt} (default: {default_value}): ").strip(); return default_value if not val_str else int(val_str)
        except ValueError: print("Invalid input. Please enter a whole number.")

def get_float_input(prompt, default_value):
    while True:
        try: val_str = input(f"{prompt} (default: {default_value:.4f}): ").strip(); return default_value if not val_str else float(val_str)
        except ValueError: print("Invalid input. Please enter a number.")

def select_genome_file(prompt_message, allow_none=False, none_option_text="None"):
    print(f"\n{prompt_message}")
    search_dirs = [GENOME_STORAGE_DIR, BEST_GENOMES_PER_GENERATION_DIR, 
                   os.path.join(GENOME_STORAGE_DIR, "final_bests"), FINETUNED_GENOME_DIR]
    collected_paths = set()
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for f_name in os.listdir(s_dir):
                if f_name.endswith(".npz"): collected_paths.add(os.path.join(s_dir, f_name))
    
    sorted_paths = sorted(list(collected_paths), key=lambda p: (os.path.dirname(p), os.path.basename(p)))
    options = []
    if allow_none: options.append((None, none_option_text))
    for path in sorted_paths: options.append((path, f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"))

    if not options and not allow_none and not sorted_paths:
         print(f"No genomes found. Please type a full path to a genome .npz file:")
         while True:
            path_input = input().strip()
            if os.path.exists(path_input) and path_input.endswith(".npz"): return path_input
            else: print("File not found or not a .npz file. Try again:")
    elif (not options and allow_none) or (not sorted_paths and allow_none and len(options) == 1 and options[0][0] is None) :
        print(f"No genomes found in monitored directories.")
        user_path = input(f"Press Enter for '{none_option_text}', or type a full path to a genome: ").strip()
        if not user_path: return None
        if os.path.exists(user_path) and user_path.endswith(".npz"): return user_path
        print("Invalid path specified. Defaulting to 'None'.")
        return None
    
    for i, (path, display_name) in enumerate(options): print(f"{i}. {display_name}")
    while True:
        raw_choice = input(f"Select by number (0-{len(options)-1}) or type full path: ").strip()
        if os.path.exists(raw_choice) and raw_choice.endswith(".npz"): return raw_choice
        try:
            choice_idx = int(raw_choice)
            if 0 <= choice_idx < len(options): return options[choice_idx][0]
            else: print("Invalid number.")
        except ValueError: print("Invalid input. Please enter a number or a valid file path.")


# --- Menu Mode Functions ---
def menu_run_manual():
    print("\n--- Manual Play Setup ---")
    opponent_genome = select_genome_file("Select AI opponent (optional):", allow_none=True, none_option_text="Random AI/Dummies")
    run_manual_simulation(opponent_genome_path=opponent_genome)

def menu_run_training():
    print("\n--- Configure Training Session ---")
    generations = get_int_input("Number of generations", DEFAULT_GENERATIONS)
    pop_size = get_int_input("Population size", DEFAULT_POPULATION_SIZE)
    elites = get_int_input("Number of elites", DEFAULT_NUM_ELITES)
    mut_sigma = get_float_input("Mutation sigma", DEFAULT_MUTATION_SIGMA)
    eval_matches = get_int_input("Evaluation matches per genome", DEFAULT_EVAL_MATCHES)
    sim_dt_chosen = get_float_input("Simulation time step (dt) for training", SIMULATION_DT)
    default_match_steps = int(MATCH_DURATION_SECONDS / sim_dt_chosen) if sim_dt_chosen > 0 else DEFAULT_MATCH_MAX_STEPS
    match_steps = get_int_input(f"Max steps per eval match (~{MATCH_DURATION_SECONDS}s)", default_match_steps)
    if input("Proceed with training? (y/n): ").strip().lower() == 'y':
        run_training_session(generations, pop_size, elites, mut_sigma, eval_matches, match_steps, sim_dt_chosen)
    else: print("Training cancelled.")

def menu_run_match(): # MODIFIED
    print("\n--- Visual Match Setup (AI vs AI) ---")
    genome1_path = select_genome_file("Choose Genome 1 (Red):", allow_none=False)
    if not genome1_path: return
    genome2_path = select_genome_file("Choose Genome 2 (Green):", allow_none=False)
    if not genome2_path: return
    if genome1_path == genome2_path: print("Warning: Same genome for both agents.")
    
    num_games = get_int_input("Number of games to play in this series", 1)
    if num_games < 1: num_games = 1
        
    run_visual_match(genome1_path, genome2_path, num_games=num_games)

def menu_run_show():
    print("\n--- Show Genome Setup ---")
    genome_path = select_genome_file("Select a genome to showcase:", allow_none=False)
    if not genome_path: return
    run_show_genome(genome_path, scenario='vs_dummies')

def menu_run_post_tournament():
    print("\n--- Post-Training Tournament Setup ---")
    visual = input("Run tournament visually? (y/n, headless is faster for scoring): ").strip().lower() == 'y'
    if visual: print("Visual tournament. You will score each match manually.")
    run_post_training_tournament(visual=visual)

def menu_run_finetune_rl():
    print("\n--- RL Fine-Tuning Setup (Self-Play) ---")
    genome_path = select_genome_file("Select genome to fine-tune:", allow_none=False)
    if not genome_path: return

    try:
        genome_to_fine_tune = persist.load_genome(genome_path)
        if not isinstance(genome_to_fine_tune, TinyNet):
            print("Error: Loaded object is not a TinyNet instance.")
            return
    except Exception as e:
        print(f"Error loading genome: {e}"); return

    num_episodes_input = get_int_input("Number of self-play episodes for RL tuning", 50)
    learning_rate_input = get_float_input("Learning rate for RL tuning", 0.001)
    
    base_name, ext_name = os.path.splitext(os.path.basename(genome_path))
    default_save_filename = f"{base_name}_rl_ft{ext_name}"
    if not os.path.exists(FINETUNED_GENOME_DIR): os.makedirs(FINETUNED_GENOME_DIR)
    default_save_path = os.path.join(FINETUNED_GENOME_DIR, default_save_filename)
    
    save_path_input = input(f"Save fine-tuned genome to (default: {default_save_path}): ").strip()
    if not save_path_input: save_path_input = default_save_path

    print(f"Starting RL fine-tuning for: {genome_path}")
    fine_tuned_genome = fine_tune_genome_with_rl(
        genome_to_fine_tune, 
        num_episodes=num_episodes_input,
        learning_rate=learning_rate_input
    )
    
    save_dir_menu = os.path.dirname(save_path_input)
    save_filename_prefix_menu = os.path.basename(save_path_input).replace(".npz","").split("_fit")[0]
    if not os.path.exists(save_dir_menu): os.makedirs(save_dir_menu)

    saved_ft_path_menu = persist.save_genome(
        fine_tuned_genome, save_filename_prefix_menu, save_dir_menu, fitness=fine_tuned_genome.fitness 
    )
    print(f"RL fine-tuned genome saved to: {saved_ft_path_menu}")
    print("Re-evaluate in arena to see performance changes.")


# --- Main Menu Display and Loop ---
def display_main_menu():
    print("\n===== Evo Arena Main Menu =====")
    print("1. Manual Play vs AI/Dummies")
    print("2. Train New AI Agents (Evolution)")
    print("3. Visual Match (AI vs AI)")
    print("4. Showcase a Trained AI Genome")
    print("5. Run Post-Training Tournament of Champions")
    print("6. Fine-Tune Genome with RL (Self-Play)")
    print("-------------------------------")
    print("0. Exit")
    print("==============================")

def main_menu_loop():
    while True:
        display_main_menu()
        choice = input("Enter your choice: ").strip()
        if choice == '1': menu_run_manual()
        elif choice == '2': menu_run_training()
        elif choice == '3': menu_run_match()
        elif choice == '4': menu_run_show()
        elif choice == '5': menu_run_post_tournament()
        elif choice == '6': menu_run_finetune_rl()
        elif choice == '0': print("Exiting Evo Arena. Goodbye!"); break
        else: print("Invalid choice, please try again.")

# --- Main Execution ---
def main():
    if not os.path.exists(FINETUNED_GENOME_DIR):
        os.makedirs(FINETUNED_GENOME_DIR)

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Evo Arena: A simple agent evolution project.")
        parser.add_argument('mode', nargs='?', default=None, 
                            choices=['manual', 'train', 'show', 'match', 'tournament', 'finetune_rl'],
                            help="Mode to run. If no mode, menu is shown.")
        # Evolution args
        parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS)
        parser.add_argument('--pop_size', type=int, default=DEFAULT_POPULATION_SIZE)
        parser.add_argument('--elites', type=int, default=DEFAULT_NUM_ELITES)
        parser.add_argument('--mut_sigma', type=float, default=DEFAULT_MUTATION_SIGMA)
        parser.add_argument('--eval_matches', type=int, default=DEFAULT_EVAL_MATCHES)
        parser.add_argument('--sim_dt', type=float, default=SIMULATION_DT)
        parser.add_argument('--match_steps', type=int, default=DEFAULT_MATCH_MAX_STEPS) # For training
        # Manual/Match/Show args
        parser.add_argument('--opponent_genome', type=str, dest='manual_opponent_genome_path')
        parser.add_argument('--g1', type=str, dest='genome1_path', help="Path to genome for AI 1 (Red)")
        parser.add_argument('--g2', type=str, dest='genome2_path', help="Path to genome for AI 2 (Green)")
        parser.add_argument('--num_games', type=int, default=1, help="Number of games for 'match' mode series.") # NEW for match
        parser.add_argument('--genome', type=str, dest='show_genome_path')
        parser.add_argument('--scenario', type=str, default='vs_dummies', choices=['vs_dummies'])
        # Tournament args
        parser.add_argument('--tournament_dir', type=str, default=BEST_GENOMES_PER_GENERATION_DIR)
        parser.add_argument('--visual_tournament', action='store_true')
        # Fine-tuning args (RL)
        parser.add_argument('--genome_path_for_finetune', type=str, help="Path to .npz genome for fine-tuning.")
        parser.add_argument('--finetune_episodes', type=int, default=50, help="Number of self-play episodes for RL fine-tuning.")
        parser.add_argument('--finetune_lr', type=float, default=0.001, help="Learning rate for RL fine-tuning.")
        parser.add_argument('--save_finetuned_path', type=str, help="Path to save fine-tuned genome (e.g., storage/genomes/finetuned/my_ft_genome.npz).")

        args = parser.parse_args()

        if args.mode is None:
            main_menu_loop()
            return

        current_match_steps_for_training = args.match_steps # For training mode
        if args.mode == 'train':
            sim_dt_is_custom = (args.sim_dt != SIMULATION_DT)
            # Match steps for training default is based on code's SIMULATION_DT
            match_steps_is_default_calc_for_training = (args.match_steps == int(MATCH_DURATION_SECONDS / SIMULATION_DT)) 
            
            if sim_dt_is_custom and match_steps_is_default_calc_for_training:
                if args.sim_dt > 0:
                    current_match_steps_for_training = int(MATCH_DURATION_SECONDS / args.sim_dt)
                    print(f"Note: --sim_dt changed for training. Adjusting --match_steps from {args.match_steps} to {current_match_steps_for_training} to maintain ~{MATCH_DURATION_SECONDS}s match duration.")
                else: 
                    print(f"Warning: Invalid --sim_dt ({args.sim_dt}) for training. Using default match_steps ({current_match_steps_for_training}).")
        
        if args.mode == 'manual':
            run_manual_simulation(opponent_genome_path=args.manual_opponent_genome_path)
        elif args.mode == 'train':
            run_training_session(args.generations, args.pop_size, args.elites, args.mut_sigma, args.eval_matches, current_match_steps_for_training, args.sim_dt)
        elif args.mode == 'match':
            if not (args.genome1_path and args.genome2_path): parser.error("'match' mode requires --g1 and --g2.")
            if not os.path.exists(args.genome1_path): parser.error(f"Genome file not found for --g1: {args.genome1_path}")
            if not os.path.exists(args.genome2_path): parser.error(f"Genome file not found for --g2: {args.genome2_path}")
            num_games_cli = args.num_games if args.num_games >=1 else 1
            run_visual_match(args.genome1_path, args.genome2_path, num_games=num_games_cli)
        elif args.mode == 'show':
            if not args.show_genome_path: parser.error("'show' mode requires --genome.")
            if not os.path.exists(args.show_genome_path): parser.error(f"Genome file not found for --genome: {args.show_genome_path}")
            run_show_genome(args.show_genome_path, scenario=args.scenario)
        elif args.mode == 'tournament':
            run_post_training_tournament(tournament_genome_dir=args.tournament_dir, visual=args.visual_tournament)
        
        elif args.mode == 'finetune_rl':
            if not args.genome_path_for_finetune: parser.error("'finetune_rl' mode requires --genome_path_for_finetune.")
            if not os.path.exists(args.genome_path_for_finetune): parser.error(f"Genome file not found for fine-tuning: {args.genome_path_for_finetune}")
            
            try:
                genome_to_ft = persist.load_genome(args.genome_path_for_finetune)
                if not isinstance(genome_to_ft, TinyNet): parser.error("Loaded object is not a TinyNet instance.")
            except Exception as e: parser.error(f"Error loading genome: {e}")

            print(f"[DEBUG CLI] Calling fine_tune_genome_with_rl with episodes: {args.finetune_episodes}, lr: {args.finetune_lr}")

            fine_tuned_g = fine_tune_genome_with_rl(
                genome_to_ft, 
                num_episodes=args.finetune_episodes,
                learning_rate=args.finetune_lr
            )
            
            save_path_ft = args.save_finetuned_path
            if not save_path_ft:
                base, ext = os.path.splitext(args.genome_path_for_finetune)
                save_path_ft = os.path.join(FINETUNED_GENOME_DIR, f"{os.path.basename(base)}_rl_ft{ext}")

            ft_save_dir = os.path.dirname(save_path_ft)
            ft_filename_prefix = os.path.basename(save_path_ft).replace(".npz","").split("_fit")[0]
            if not os.path.exists(ft_save_dir): os.makedirs(ft_save_dir)

            saved_ft_final_path = persist.save_genome(fine_tuned_g, ft_filename_prefix, ft_save_dir, fitness=fine_tuned_g.fitness)
            print(f"RL fine-tuned genome saved to: {saved_ft_final_path}")
            print("Re-evaluate performance in the arena.")
    else:
        main_menu_loop()

if __name__ == '__main__':
    main()