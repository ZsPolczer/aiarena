# evo_arena/main.py
import pygame
import argparse # For CLI later
from arena.arena import Arena
from agents.body import AgentBody
from agents.brain import TinyNet # Will be needed for AI agents
from ui.viewer import Viewer
# from storage import persist # For loading/saving brains

# --- Configuration ---
ARENA_WIDTH = 800
ARENA_HEIGHT = 800
FPS = 50 

AGENT_BASE_SPEED = 200
AGENT_ROTATION_SPEED_DPS = 180
AGENT_RADIUS = 15
WALL_BOUNCE_LOSS_FACTOR = 0.9

MANUAL_AGENT_COLOR = (0, 150, 255)
DUMMY_AGENT_COLOR = (255, 100, 0)
AI_AGENT_COLOR = (0, 200, 50) # Green for AI

# --- New Weapon Defaults ---
WEAPON_RANGE = 150            # Increased from 80
WEAPON_ARC_DEG = 90           # Reduced from 120 (narrower cone)
WEAPON_COOLDOWN_TIME = 0.6
WEAPON_DAMAGE = 25            # Increased from 10

def run_manual_simulation():
    """Runs the simulation with one manual agent and one dummy."""
    game_arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, wall_bounce_loss_factor=WALL_BOUNCE_LOSS_FACTOR)

    manual_agent = AgentBody(
        x=100, y=ARENA_HEIGHT / 2,
        angle_deg=0,
        base_speed=AGENT_BASE_SPEED,
        rotation_speed_dps=AGENT_ROTATION_SPEED_DPS,
        radius=AGENT_RADIUS,
        color=MANUAL_AGENT_COLOR,
        agent_id="player",
        team_id=1,
        hp=100,
        brain=None, # Manual control
        weapon_range=WEAPON_RANGE,             # Using new constant
        weapon_arc_deg=WEAPON_ARC_DEG,         # Using new constant
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME,
        weapon_damage=WEAPON_DAMAGE            # Using new constant
    )
    game_arena.add_agent(manual_agent)

    dummy_target = AgentBody(
        x=ARENA_WIDTH - 100, y=ARENA_HEIGHT / 2,
        angle_deg=180,
        base_speed=0,
        rotation_speed_dps=0,
        radius=AGENT_RADIUS + 5,
        color=DUMMY_AGENT_COLOR,
        agent_id="dummy",
        team_id=2,
        hp=200, # Dummy has more HP
        is_dummy=True,
        brain=None,
        weapon_range=0, # Dummy has no weapon
        weapon_arc_deg=0,
        weapon_cooldown_time=999,
        weapon_damage=0
    )
    game_arena.add_agent(dummy_target)
    
    # Add a stationary AI agent for testing targeting (optional)
    # stationary_ai_brain = TinyNet() # Default random brain
    # stationary_ai_agent = AgentBody(
    #     x=ARENA_WIDTH / 2, y=ARENA_HEIGHT - 100,
    #     angle_deg=-90, base_speed=AGENT_BASE_SPEED, rotation_speed_dps=AGENT_ROTATION_SPEED_DPS,
    #     radius=AGENT_RADIUS, color=AI_AGENT_COLOR, agent_id="ai_stationary", team_id=2, hp=100,
    #     brain=stationary_ai_brain, # Give it a brain
    #     weapon_range=WEAPON_RANGE, weapon_arc_deg=WEAPON_ARC_DEG,
    #     weapon_cooldown_time=WEAPON_COOLDOWN_TIME, weapon_damage=WEAPON_DAMAGE
    # )
    # game_arena.add_agent(stationary_ai_agent)


    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT, game_arena)
    game_viewer.run_simulation_loop(FPS, manual_agent_id="player") # Pass arena via constructor

# --- Main execution flow ---
def main():
    # For now, directly call the manual simulation.
    # Later, we'll use argparse to select modes like 'train', 'show', 'match'.
    parser = argparse.ArgumentParser(description="Evo Arena: A simple agent evolution project.")
    parser.add_argument('mode', nargs='?', default='manual', choices=['manual', 'train', 'show', 'match'],
                        help="Mode to run: 'manual' (default), 'train', 'show' a replay, or 'match' agents.")
    # Add more arguments for specific modes later

    args = parser.parse_args()

    if args.mode == 'manual':
        run_manual_simulation()
    # elif args.mode == 'train':
    #     run_training_session() # To be implemented
    else:
        print(f"Mode '{args.mode}' not fully implemented yet. Running manual simulation.")
        run_manual_simulation()


if __name__ == '__main__':
    main()