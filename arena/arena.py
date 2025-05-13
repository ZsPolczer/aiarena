# evo_arena/arena/arena.py
import pygame
import math # Needed for atan2, degrees, radians, sqrt, cos, sin

class Arena:
    def __init__(self, width, height, wall_bounce_loss_factor=0.9):
        self.width = width
        self.height = height
        self.agents = []
        self.wall_bounce_loss_factor = wall_bounce_loss_factor
        self.game_time = 0.0 # For match timeout later

    def add_agent(self, agent):
        self.agents.append(agent)

    def get_alive_agents(self):
        return [agent for agent in self.agents if agent.is_alive()]

    def update(self, dt):
        self.game_time += dt
        
        # Get all agents that are currently alive
        # Process agent updates (movement, brain logic) first
        # Pass all agents for sensory input, but only alive ones are relevant for interaction
        current_all_agents_state = self.agents # Pass all, AgentBody.get_inputs will filter by .is_alive()
        
        for agent in self.agents:
            if agent.is_alive():
                 # Agent update (includes brain processing, action decisions, internal cooldowns)
                agent.update(dt, self.width, self.height, current_all_agents_state)
            else:
                agent.vx = 0 # Stop dead agents
                agent.vy = 0

        # Process firing and hit detection
        for idx, firing_agent in enumerate(self.agents):
            if not firing_agent.is_alive() or not firing_agent.is_firing_command:
                continue

            # If agent is commanding to fire AND its cooldown allowed it (is_firing_command is true)
            # The cooldown reset happens in AgentBody.update based on NN output or manual input.
            # So, if is_firing_command is true, it means an attempt to fire is made this tick.
            
            # For visualization, we might want to store that a shot was fired
            # firing_agent.shot_this_tick = True # Add this attribute to AgentBody if needed for drawing

            # Check for hits on other agents
            for target_idx, target_agent in enumerate(self.agents):
                if idx == target_idx or not target_agent.is_alive(): # Cannot shoot self or dead agents
                    continue

                # Check for friendly fire: by default, only hit enemies
                if firing_agent.team_id == target_agent.team_id: # Simple friendly fire check
                    # For some game modes, friendly fire might be enabled.
                    # For now, assume no friendly fire unless specified.
                    continue 

                # 1. Distance Check
                dx = target_agent.x - firing_agent.x
                dy = target_agent.y - firing_agent.y
                distance_sq = dx*dx + dy*dy # Use squared distance to avoid sqrt until necessary
                
                # Consider agent radii for effective range
                # effective_range = firing_agent.weapon_range + target_agent.radius 
                effective_range = firing_agent.weapon_range # Spec says "max range = 80", assume from center to center for simplicity
                                                          # Or edge of firing agent to center/edge of target.
                                                          # Let's use center-to-center for now.

                if distance_sq <= effective_range * effective_range:
                    distance = math.sqrt(distance_sq) # Now do sqrt
                    if distance == 0: continue # Avoid division by zero if agents are exactly on top of each other

                    # 2. Angle Check (120° cone = +/- 60° from agent's direction)
                    angle_to_target_rad = math.atan2(dy, dx)
                    agent_facing_rad = math.radians(firing_agent.angle_deg)
                    
                    # Relative angle of target with respect to agent's facing direction
                    relative_angle_rad = angle_to_target_rad - agent_facing_rad
                    
                    # Normalize to [-pi, pi]
                    relative_angle_rad = (relative_angle_rad + math.pi) % (2 * math.pi) - math.pi
                    
                    weapon_half_arc_rad = math.radians(firing_agent.weapon_arc_deg / 2.0)
                    
                    if abs(relative_angle_rad) <= weapon_half_arc_rad:
                        # HIT!
                        print(f"HIT! Agent {firing_agent.agent_id} (Team {firing_agent.team_id}) shot Agent {target_agent.agent_id} (Team {target_agent.team_id})")
                        target_agent.take_damage(firing_agent.weapon_damage)
                        # Potentially break here if weapon only hits one target per shot
                        # The spec doesn't specify (e.g. piercing), assume one hit per target in cone.
                        # If a single shot can hit multiple, don't break. Current loop handles this.

            # Reset the command flag for the firing agent after processing its shot for this tick
            # This is important if the brain/manual input keeps the fire signal high.
            # The agent's internal cooldown (weapon_cooldown_timer) is the primary gate.
            # is_firing_command is set by agent if cooldown permits.
            # If we reset it here, it means each fire *output* from NN leads to one check.
            # This seems reasonable.
            # firing_agent.is_firing_command = False # This might be too aggressive if agent already handles cooldown.
            # The agent already handles its cooldown and only sets is_firing_command if it CAN fire.
            # So, if is_firing_command is true, it means it has passed its cooldown check for this tick.
            # No need to reset it here, it will be re-evaluated by agent logic next tick.

        # Physics and Wall Collisions
        for agent in self.agents:
            if not agent.is_alive():
                continue # Skip physics for dead agents

            # Agent has already updated its vx, vy and potentially x, y in its own agent.update()
            # Now apply wall collisions based on the new prospective position (which is already its current x,y)

            # Store pre-collision speed for bounce reduction calculation
            # speed_before_collision = math.sqrt(agent.vx**2 + agent.vy**2) # Not directly used by spec for speed loss.

            collided_x = False
            collided_y = False

            # Left wall
            if agent.x - agent.radius < 0:
                agent.x = agent.radius
                agent.vx *= -self.wall_bounce_loss_factor
                collided_x = True
            # Right wall
            elif agent.x + agent.radius > self.width:
                agent.x = self.width - agent.radius
                agent.vx *= -self.wall_bounce_loss_factor
                collided_x = True
            
            # Top wall
            if agent.y - agent.radius < 0:
                agent.y = agent.radius
                agent.vy *= -self.wall_bounce_loss_factor
                collided_y = True
            # Bottom wall
            elif agent.y + agent.radius > self.height:
                agent.y = self.height - agent.radius
                agent.vy *= -self.wall_bounce_loss_factor
                collided_y = True
            
            # If collision occurred, apply 10% speed loss to the current_speed of the agent
            # The spec says "10% speed loss". This implies the magnitude of velocity.
            # Agent's current_speed might be a good proxy if it reflects the intended speed.
            # Or, more directly, scale the magnitude of (vx, vy).
            if (collided_x or collided_y) and hasattr(agent, 'base_speed'): # Check if it's a controllable agent
                # Reduce overall speed. The agent's next update() will re-calculate vx,vy based on new (potentially reduced) base_speed or current_speed.
                # This is tricky because "teleport-step" velocity is set by actions each frame.
                # A simple way to interpret "10% speed loss" is on the resulting velocity components *after* bounce.
                # This is already done by agent.vx *= -self.wall_bounce_loss_factor.
                # If it means a more persistent loss of thrusting capability, that's different.
                # The current implementation of agent.vx *= -factor effectively reduces the speed *for that bounce*.
                # If agent applies full thrust next frame, it goes back to full speed.
                # For now, let's stick to the current bounce logic. It reduces speed for the bounce reflection.
                pass


    def draw_bounds(self, screen):
        pygame.draw.rect(screen, (50, 50, 50), (0, 0, self.width, self.height), 2)

    def check_match_end_conditions(self):
        """
        Checks if the match should end.
        Returns: (is_over, winner_team_id_or_None_for_draw, message)
        """
        alive_agents = self.get_alive_agents()

        if not alive_agents:
            return True, None, "All agents eliminated (Draw or mutual destruction)"

        # Check for timeout (60s)
        if self.game_time >= 60.0:
            # Determine winner by HP or other tie-breaking rules if desired.
            # For now, timeout is a draw as per spec unless one team is standing.
            # "last team standing or 60 s timeout (draw)."
            # If timeout, and multiple teams still alive, it's a draw.
            # If timeout, and only one team alive, that team wins.
            
            teams_alive = set()
            for agent in alive_agents:
                teams_alive.add(agent.team_id)
            
            if len(teams_alive) > 1:
                return True, None, f"Timeout after {self.game_time:.1f}s. Draw."
            elif len(teams_alive) == 1:
                winner_team_id = list(teams_alive)[0]
                return True, winner_team_id, f"Timeout. Team {winner_team_id} wins (last standing)."
            else: # Should not happen if alive_agents is not empty
                return True, None, "Timeout. Draw (no teams identified)."


        # Check for last team standing
        if alive_agents:
            first_agent_team_id = alive_agents[0].team_id
            all_same_team = True
            for agent in alive_agents:
                if agent.team_id != first_agent_team_id:
                    all_same_team = False
                    break
            if all_same_team:
                return True, first_agent_team_id, f"Team {first_agent_team_id} is the last one standing!"
        
        return False, None, "Match ongoing"

    def reset(self):
        """Resets the arena for a new match."""
        self.game_time = 0.0
        # Agents themselves need to be reset (HP, position, etc.)
        # This is usually handled by recreating agents or calling a reset method on them.
        # For now, assuming agents are re-added/re-initialized by the evolution loop.
        # If running standalone matches, we might need an agent.reset()
        for agent in self.agents:
            # This is a placeholder; proper reset might involve re-initializing them to start positions/HP
            # For evolution, new agents (or reset ones) will be placed.
            # For now, if we call arena.reset(), it mainly resets game_time.
            # Agent state should be handled by whatever system starts the match.
            if hasattr(agent, 'initial_x') and hasattr(agent, 'initial_y') and hasattr(agent, 'initial_angle_deg'):
                agent.x = agent.initial_x
                agent.y = agent.initial_y
                agent.angle_deg = agent.initial_angle_deg
                agent.hp = agent.max_hp
                agent.weapon_cooldown_timer = 0.0
                agent.vx = 0.0
                agent.vy = 0.0
                agent.is_firing_command = False
            else: # Fallback if initial states not stored
                agent.hp = agent.max_hp 
                agent.weapon_cooldown_timer = 0.0