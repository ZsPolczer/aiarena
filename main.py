# evo_arena/main.py
import pygame # For key constants
# If you use TinyNet directly in main for testing, you might need this:
# from agents.brain import TinyNet 
from arena.arena import Arena
from agents.body import AgentBody
from ui.viewer import Viewer
# from storage import persist # For loading/saving brains later

# --- Configuration ---
ARENA_WIDTH = 800
ARENA_HEIGHT = 800
FPS = 50 # Game clock frequency, as per spec

AGENT_BASE_SPEED = 200  # "meters" (pixels) per second
AGENT_ROTATION_SPEED_DPS = 180  # degrees per second # <<< Ensure this constant name is clear
AGENT_RADIUS = 15
WALL_BOUNCE_LOSS_FACTOR = 0.9 # 10% speed loss

MANUAL_AGENT_COLOR = (0, 150, 255) # Blueish
DUMMY_AGENT_COLOR = (255, 100, 0) # Orangish

# Weapon defaults (can be overridden per agent if needed)
WEAPON_RANGE = 80
WEAPON_ARC_DEG = 120
WEAPON_COOLDOWN_TIME = 0.6 # seconds
WEAPON_DAMAGE = 10

def main():
    # Initialize Arena
    game_arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, WALL_BOUNCE_LOSS_FACTOR)

    # Initialize Agents
    manual_agent = AgentBody(
        x=100, y=ARENA_HEIGHT / 2,
        angle_deg=0,
        base_speed=AGENT_BASE_SPEED,
        rotation_speed_dps=AGENT_ROTATION_SPEED_DPS, # <<< CORRECTED HERE
        radius=AGENT_RADIUS,
        color=MANUAL_AGENT_COLOR,
        agent_id="player",
        team_id=1, # Example team ID
        hp=100,
        brain=None, # No brain for manual agent
        weapon_range=WEAPON_RANGE,
        weapon_arc_deg=WEAPON_ARC_DEG,
        weapon_cooldown_time=WEAPON_COOLDOWN_TIME,
        weapon_damage=WEAPON_DAMAGE
    )
    game_arena.add_agent(manual_agent)

    dummy_target = AgentBody(
        x=ARENA_WIDTH - 100, y=ARENA_HEIGHT / 2,
        angle_deg=180,
        base_speed=0, # Dummy doesn't move on its own
        rotation_speed_dps=0, # <<< CORRECTED HERE
        radius=AGENT_RADIUS + 5, # Slightly larger
        color=DUMMY_AGENT_COLOR,
        agent_id="dummy",
        team_id=2, # Example different team ID
        hp=200, # Dummy might have more HP
        is_dummy=True,
        brain=None,
        weapon_range=0, # Dummy has no weapon
        weapon_arc_deg=0,
        weapon_cooldown_time=999,
        weapon_damage=0
    )
    game_arena.add_agent(dummy_target)

    # Initialize Viewer
    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT)

    # Start the manual control loop
    # The viewer loop will now need to pass more info to arena.update if brains are active
    # For now, manual agent doesn't use brain, so its update doesn't strictly need all_agents etc.
    game_viewer.run_manual_loop(game_arena, manual_agent, FPS)

if __name__ == '__main__':
    main()