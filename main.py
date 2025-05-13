# evo_arena/main.py
import pygame # For key constants
from arena.arena import Arena
from agents.body import AgentBody
from ui.viewer import Viewer

# --- Configuration ---
ARENA_WIDTH = 800
ARENA_HEIGHT = 800
FPS = 50 # Game clock frequency, as per spec

AGENT_BASE_SPEED = 200  # "meters" (pixels) per second
AGENT_ROTATION_SPEED_DPS = 180  # degrees per second
AGENT_RADIUS = 15
WALL_BOUNCE_LOSS_FACTOR = 0.9 # 10% speed loss

MANUAL_AGENT_COLOR = (0, 150, 255) # Blueish
DUMMY_AGENT_COLOR = (255, 100, 0) # Orangish

def main():
    # Initialize Arena
    game_arena = Arena(ARENA_WIDTH, ARENA_HEIGHT, WALL_BOUNCE_LOSS_FACTOR)

    # Initialize Agents
    manual_agent = AgentBody(
        x=100, y=ARENA_HEIGHT / 2,
        angle_deg=0,
        base_speed=AGENT_BASE_SPEED,
        rotation_speed=AGENT_ROTATION_SPEED_DPS,
        radius=AGENT_RADIUS,
        color=MANUAL_AGENT_COLOR,
        is_dummy=False,
        agent_id="player"
    )
    game_arena.add_agent(manual_agent)

    dummy_target = AgentBody(
        x=ARENA_WIDTH - 100, y=ARENA_HEIGHT / 2,
        angle_deg=180,
        base_speed=0, # Dummy doesn't move on its own
        rotation_speed=0,
        radius=AGENT_RADIUS + 5, # Slightly larger
        color=DUMMY_AGENT_COLOR,
        is_dummy=True,
        agent_id="dummy"
    )
    game_arena.add_agent(dummy_target)

    # Initialize Viewer
    game_viewer = Viewer(ARENA_WIDTH, ARENA_HEIGHT)

    # Start the manual control loop
    game_viewer.run_manual_loop(game_arena, manual_agent, FPS)

if __name__ == '__main__':
    main()