# evo_arena/arena/arena.py
import pygame # Only for Rect if used in drawing, not strictly needed for headless logic
import math
from agents.body import AgentBody # Added for run_match
from agents.brain import TinyNet # Added for type hinting in run_match

# Default parameters for agents created in run_match
# These could be moved to a config file or passed as arguments for more flexibility
DEFAULT_AGENT_HP = 100
DEFAULT_AGENT_RADIUS = 15 
DEFAULT_AGENT_BASE_SPEED = 150 
DEFAULT_AGENT_ROTATION_SPEED_DPS = 180 
DEFAULT_WEAPON_RANGE = 150 
DEFAULT_WEAPON_ARC_DEG = 90 
DEFAULT_WEAPON_COOLDOWN_TIME = 0.6 
DEFAULT_WEAPON_DAMAGE = 25 
# DEFAULT_FPS is used by caller to determine dt and max_steps


class Arena:
    def __init__(self, width, height, wall_bounce_loss_factor=0.9):
        self.width = width
        self.height = height
        self.agents = [] # List of AgentBody objects
        self.wall_bounce_loss_factor = wall_bounce_loss_factor
        self.game_time = 0.0 # Current time in the match (seconds)

    def add_agent(self, agent):
        # Ensure agent has initial state stored for potential resets
        if not hasattr(agent, 'initial_x'):
            agent.initial_x = agent.x
        if not hasattr(agent, 'initial_y'):
            agent.initial_y = agent.y
        if not hasattr(agent, 'initial_angle_deg'):
            agent.initial_angle_deg = agent.angle_deg
        if not hasattr(agent, 'max_hp_initial'): # Store the HP it's created with as its "full" HP for resets
            agent.max_hp_initial = agent.max_hp
        
        self.agents.append(agent)

    def get_alive_agents(self):
        return [agent for agent in self.agents if agent.is_alive()]
    
    def get_agent_by_id(self, agent_id):
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def update(self, dt):
        self.game_time += dt
        
        current_all_agents_state = list(self.agents) # Pass a stable list for this tick's decisions
        
        for agent in self.agents:
            if agent.is_alive():
                agent.update(dt, self.width, self.height, current_all_agents_state)
            else:
                agent.vx = 0 
                agent.vy = 0
                agent.is_firing_command = False


        # Process firing and hit detection
        for idx, firing_agent in enumerate(self.agents):
            if not firing_agent.is_alive() or not firing_agent.is_firing_command:
                continue
            
            for target_idx, target_agent in enumerate(self.agents):
                if idx == target_idx or not target_agent.is_alive():
                    continue
                if firing_agent.team_id == target_agent.team_id and firing_agent.team_id != 0: # No friendly fire unless team_id is 0 (e.g. FFA)
                    continue 

                dx = target_agent.x - firing_agent.x
                dy = target_agent.y - firing_agent.y
                distance_sq = dx*dx + dy*dy
                
                effective_range = firing_agent.weapon_range 

                if distance_sq <= effective_range * effective_range:
                    if distance_sq == 0: continue 

                    angle_to_target_rad = math.atan2(dy, dx)
                    agent_facing_rad = math.radians(firing_agent.angle_deg)
                    relative_angle_rad = (angle_to_target_rad - agent_facing_rad + math.pi) % (2 * math.pi) - math.pi
                    weapon_half_arc_rad = math.radians(firing_agent.weapon_arc_deg / 2.0)
                    
                    if abs(relative_angle_rad) <= weapon_half_arc_rad:
                        target_agent.take_damage(firing_agent.weapon_damage)
            
        # Physics and Wall Collisions
        for agent in self.agents:
            if not agent.is_alive():
                continue

            collided_x = False
            collided_y = False

            if agent.x - agent.radius < 0:
                agent.x = agent.radius
                agent.vx *= -self.wall_bounce_loss_factor
                collided_x = True
            elif agent.x + agent.radius > self.width:
                agent.x = self.width - agent.radius
                agent.vx *= -self.wall_bounce_loss_factor
                collided_x = True
            
            if agent.y - agent.radius < 0:
                agent.y = agent.radius
                agent.vy *= -self.wall_bounce_loss_factor
                collided_y = True
            elif agent.y + agent.radius > self.height:
                agent.y = self.height - agent.radius
                agent.vy *= -self.wall_bounce_loss_factor
                collided_y = True

    def draw_bounds(self, screen): 
        if 'pygame' in globals() and screen is not None: # Check if pygame is available and screen is provided
             pygame.draw.rect(screen, (50, 50, 50), (0, 0, self.width, self.height), 2)

    def check_match_end_conditions(self, max_duration_seconds=60.0):
        """
        Checks if the match should end.
        Returns: (is_over, winner_team_id_or_None_for_draw, message_string)
        """
        alive_agents = self.get_alive_agents()

        if not alive_agents:
            return True, None, "All agents eliminated"

        # Check for timeout first
        if self.game_time >= max_duration_seconds:
            teams_alive_count = {} 
            for agent in alive_agents:
                teams_alive_count[agent.team_id] = teams_alive_count.get(agent.team_id, 0) + 1
            
            if len(teams_alive_count) > 1:
                return True, None, f"Timeout after {self.game_time:.1f}s. Draw (multiple teams alive)."
            elif len(teams_alive_count) == 1:
                winner_team_id = list(teams_alive_count.keys())[0]
                return True, winner_team_id, f"Timeout. Team {winner_team_id} wins (last standing)."
            else: # Should not happen if alive_agents is not empty
                 return True, None, f"Timeout after {self.game_time:.1f}s. No teams identified among survivors (Error)."


        # Check for last team standing (if not timed out yet)
        teams_present = set(agent.team_id for agent in alive_agents)
        if len(teams_present) == 1:
            winner_team_id = list(teams_present)[0]
            return True, winner_team_id, f"Team {winner_team_id} is the last one standing!"
        
        return False, None, "Match ongoing"

    def reset_arena_and_agents(self):
        """Resets arena time and all agents currently in the arena to their initial states."""
        self.game_time = 0.0
        for agent in self.agents:
            if hasattr(agent, 'reset_state') and \
               hasattr(agent, 'initial_x') and \
               hasattr(agent, 'initial_y') and \
               hasattr(agent, 'initial_angle_deg'):
                
                initial_hp_to_reset = agent.max_hp_initial if hasattr(agent, 'max_hp_initial') else agent.max_hp
                agent.reset_state(agent.initial_x, agent.initial_y, agent.initial_angle_deg, initial_hp_to_reset)
            else:
                print(f"Warning: Agent {agent.agent_id} might not be fully reset. Missing initial state attributes or reset_state method.")
                # Basic fallback if full reset isn't possible
                agent.hp = agent.max_hp if not hasattr(agent, 'max_hp_initial') else agent.max_hp_initial
                agent.weapon_cooldown_timer = 0.0
                agent.vx = 0.0
                agent.vy = 0.0
                agent.is_firing_command = False


    def run_match(self, agent_configs, max_duration_steps, dt):
        """
        Runs a single headless match between agents defined by agent_configs.
        agent_configs: A list of dictionaries, each specifying an agent.
                       Example: [{'brain': brain_obj1, 'team_id': 1, 'agent_id': 'evo_1', 
                                   'start_pos': (x,y,angle)}, ...]
        max_duration_steps: Total number of simulation steps for the match.
        dt: Time delta for each step (e.g., 1.0 / FPS).
        
        Returns a dictionary with match results.
        """
        self.agents = [] # Clear previous agents
        self.game_time = 0.0

        # Create and add agents based on configs
        for i, config in enumerate(agent_configs):
            # Define default start positions if not provided, e.g. for 1v1
            default_x = 100 if i == 0 else self.width - 100
            default_y = self.height / 2
            default_angle = 0 if i == 0 else 180
            
            start_x, start_y, start_angle = config.get('start_pos', (default_x, default_y, default_angle))
            
            agent = AgentBody(
                x=start_x, y=start_y, angle_deg=start_angle,
                base_speed=config.get('base_speed', DEFAULT_AGENT_BASE_SPEED),
                rotation_speed_dps=config.get('rotation_speed_dps', DEFAULT_AGENT_ROTATION_SPEED_DPS),
                radius=config.get('radius', DEFAULT_AGENT_RADIUS),
                color=config.get('color', (50,50,50)), 
                agent_id=config.get('agent_id', f"match_agent_{i+1}"),
                team_id=config.get('team_id', i+1), 
                hp=config.get('hp', DEFAULT_AGENT_HP),
                brain=config.get('brain'), 
                weapon_range=config.get('weapon_range', DEFAULT_WEAPON_RANGE),
                weapon_arc_deg=config.get('weapon_arc_deg', DEFAULT_WEAPON_ARC_DEG),
                weapon_cooldown_time=config.get('weapon_cooldown_time', DEFAULT_WEAPON_COOLDOWN_TIME),
                weapon_damage=config.get('weapon_damage', DEFAULT_WEAPON_DAMAGE)
            )
            self.add_agent(agent) # add_agent now also sets initial_x/y/angle/hp_max_initial

        # Simulation loop
        match_over = False
        winner_team_id = None
        end_message = "Match did not conclude properly." # Default message
        actual_steps = 0

        for step in range(max_duration_steps):
            actual_steps = step + 1
            self.update(dt) 

            max_duration_seconds = max_duration_steps * dt
            match_over, winner_team_id, end_message = self.check_match_end_conditions(max_duration_seconds=max_duration_seconds)
            
            if match_over:
                break 
        
        # If loop finished by exhausting steps, ensure end conditions reflect timeout if applicable
        if not match_over:
            max_duration_seconds = max_duration_steps * dt # Recalculate for clarity
            # This final check essentially re-evaluates based on the state after the last step, mostly for timeout.
            is_final_over, final_winner_id, final_message = self.check_match_end_conditions(max_duration_seconds=max_duration_seconds)
            if is_final_over : # If the match is indeed over by timeout (or other condition met on last step)
                winner_team_id = final_winner_id
                end_message = final_message
            else: # This case means max_duration_steps were run, but no win/loss/draw condition was met (e.g., multiple teams still strong)
                  # It should ideally be caught by check_match_end_conditions as a timeout draw.
                  # If check_match_end_conditions says "Match ongoing", but we are out of steps, it's a timeout.
                winner_team_id = None # Explicitly a draw due to step limit
                end_message = f"Max steps ({max_duration_steps}) reached. Considered Draw."

        
        final_agents_state = []
        for agent in self.agents:
            final_agents_state.append({
                'agent_id': agent.agent_id,
                'team_id': agent.team_id,
                'hp': agent.hp,
                'is_alive': agent.is_alive()
            })

        return {
            'winner_team_id': winner_team_id,
            'duration_steps': actual_steps, 
            'game_time_at_end': round(self.game_time, 3),
            'end_message': end_message,
            'agents_final_state': final_agents_state
        }