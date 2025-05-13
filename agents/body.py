# evo_arena/agents/body.py
import pygame
import math
import numpy as np # For brain inputs

# Assuming TinyNet might be passed in, or loaded
# from agents.brain import TinyNet # Not strictly needed here if brain is passed as object

class AgentBody:
    def __init__(self, x, y, angle_deg, base_speed, rotation_speed_dps,
                 radius=15, color=(0, 0, 255), agent_id="agent", team_id=0,
                 hp=100, brain=None, is_dummy=False,
                 weapon_range=80, weapon_arc_deg=120, weapon_cooldown_time=0.6, weapon_damage=10):
        
        self.agent_id = str(agent_id)
        self.team_id = int(team_id)
        self.x = float(x)
        self.y = float(y)
        self.angle_deg = float(angle_deg)
        self.radius = int(radius)
        self.color = color
        self.is_dummy = is_dummy
        self.brain = brain # This will be a TinyNet instance or None

        self.base_speed = float(base_speed)
        self.rotation_speed_dps = float(rotation_speed_dps) # Degrees per second for turning
        
        self.vx = 0.0
        self.vy = 0.0
        self.current_speed = 0.0 # Current forward/backward speed component

        self.max_hp = float(hp)
        self.hp = float(hp)
        
        # Weapon attributes
        self.weapon_range = float(weapon_range)
        self.weapon_arc_deg = float(weapon_arc_deg) # Centered on agent's forward direction
        self.weapon_cooldown_time = float(weapon_cooldown_time) # Seconds
        self.weapon_damage = int(weapon_damage)
        self.weapon_cooldown_timer = 0.0 # Counts down to 0 when ready
        self.is_firing_command = False # Set by brain/manual control, processed in arena

        # For manual control state (if brain is None)
        self._manual_thrust_forward = False
        self._manual_thrust_backward = False
        self._manual_rotate_left = False
        self._manual_rotate_right = False
        self._manual_fire = False
        
        # Clamping values for NN inputs/outputs if needed
        self.max_abs_velocity_component = self.base_speed # For normalizing velocity inputs

    def is_alive(self):
        return self.hp > 0

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp < 0:
            self.hp = 0
            print(f"Agent {self.agent_id} defeated.")

    def manual_control(self, keys):
        if self.is_dummy or self.brain is not None: # Manual control only if no brain
            return

        self._manual_thrust_forward = keys[pygame.K_UP]
        self._manual_thrust_backward = keys[pygame.K_DOWN]
        self._manual_rotate_left = keys[pygame.K_LEFT]
        self._manual_rotate_right = keys[pygame.K_RIGHT]
        self._manual_fire = keys[pygame.K_SPACE] # Example: Space to fire

    def get_inputs(self, arena_width, arena_height, all_agents):
        """
        Generates the 14 input values for the neural network, normalized to [-1, 1].
        """
        inputs = np.zeros(14) # 14 inputs as per spec

        # Helper for normalization
        def normalize(value, min_val, max_val):
            if max_val == min_val: return 0.0 # Avoid division by zero
            return np.clip(2 * (value - min_val) / (max_val - min_val) - 1, -1.0, 1.0)

        # Helper to find nearest agent (enemy or ally)
        def find_nearest_agent_in_list(target_list):
            nearest_dist = float('inf')
            nearest_agent_obj = None
            for other_agent in target_list:
                if other_agent is self or not other_agent.is_alive():
                    continue
                dx = other_agent.x - self.x
                dy = other_agent.y - self.y
                dist = math.sqrt(dx**2 + dy**2) - self.radius - other_agent.radius # Edge to edge
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_agent_obj = other_agent
            return nearest_agent_obj, nearest_dist

        # 1. Forward distance to nearest enemy
        # 2. Bearing to that enemy (sin θ, cos θ)
        enemies = [a for a in all_agents if a.team_id != self.team_id and a.is_alive()]
        nearest_enemy, enemy_dist = find_nearest_agent_in_list(enemies)
        
        max_view_dist = math.sqrt(arena_width**2 + arena_height**2) # Diagonal of arena

        if nearest_enemy:
            inputs[0] = normalize(enemy_dist, 0, max_view_dist) # Normalize distance
            
            dx_enemy = nearest_enemy.x - self.x
            dy_enemy = nearest_enemy.y - self.y
            angle_to_enemy_rad = math.atan2(dy_enemy, dx_enemy)
            
            # Bearing relative to agent's current orientation
            # Angle of agent: self.angle_deg (convert to rad)
            # Relative angle = angle_to_enemy - agent_angle
            relative_angle_rad = angle_to_enemy_rad - math.radians(self.angle_deg)
            # Normalize angle to [-pi, pi]
            relative_angle_rad = (relative_angle_rad + math.pi) % (2 * math.pi) - math.pi
            
            inputs[1] = math.sin(relative_angle_rad) # Already in [-1, 1]
            inputs[2] = math.cos(relative_angle_rad) # Already in [-1, 1]
        else:
            inputs[0] = 1.0 # No enemy in sight (or -1.0, spec says +/-1)
            inputs[1] = 0.0 # No bearing if no enemy
            inputs[2] = 0.0 

        # 3. Forward distance to nearest ally
        # 4. Bearing to that ally (sin θ, cos θ)
        allies = [a for a in all_agents if a.team_id == self.team_id and a is not self and a.is_alive()]
        nearest_ally, ally_dist = find_nearest_agent_in_list(allies)

        if nearest_ally:
            inputs[3] = normalize(ally_dist, 0, max_view_dist)
            
            dx_ally = nearest_ally.x - self.x
            dy_ally = nearest_ally.y - self.y
            angle_to_ally_rad = math.atan2(dy_ally, dx_ally)
            relative_angle_ally_rad = angle_to_ally_rad - math.radians(self.angle_deg)
            relative_angle_ally_rad = (relative_angle_ally_rad + math.pi) % (2 * math.pi) - math.pi
            
            inputs[4] = math.sin(relative_angle_ally_rad)
            inputs[5] = math.cos(relative_angle_ally_rad)
        else:
            inputs[3] = 1.0 # No ally in sight
            inputs[4] = 0.0
            inputs[5] = 0.0

        # 5. Own health / 100 (normalized to [0,1], spec says [-1,1], so scale if max_hp is 100)
        inputs[6] = normalize(self.hp, 0, self.max_hp) # Will be [-1,1] if hp can be 0 to max_hp

        # 6. Weapon ready? (1 = yes, –1 = cooling)
        inputs[7] = 1.0 if self.weapon_cooldown_timer <= 0 else -1.0

        # 7. x-velocity, y-velocity (clamped and normalized)
        # Max possible speed is base_speed for now.
        # These are world-frame velocities.
        inputs[8] = normalize(self.vx, -self.max_abs_velocity_component, self.max_abs_velocity_component)
        inputs[9] = normalize(self.vy, -self.max_abs_velocity_component, self.max_abs_velocity_component)
        
        # What are inputs 10, 11, 12? Spec shows 14 total.
        # My list:
        # 0: enemy_dist
        # 1: enemy_bearing_sin
        # 2: enemy_bearing_cos
        # 3: ally_dist
        # 4: ally_bearing_sin
        # 5: ally_bearing_cos
        # 6: health
        # 7: weapon_ready
        # 8: vx
        # 9: vy
        # --> This is 10 inputs. The spec image shows 14 inputs total.
        # Let's check the image again: "Forward distance", "Bearing (sin, cos)", "Forward distance (ally)", "Bearing (ally) (sin,cos)" = 6
        # "Own health" = 1
        # "Weapon ready" = 1
        # "x-velocity, y-velocity" = 2
        # "Bias neuron = 1" = 1
        # Total = 6+1+1+2+1 = 11.
        # The TinyNet code has `(16, 14)` for `w_in`. So 14 inputs.
        # "Inputs (all normalised to [-1, 1]) – 14 numbers total:"
        # 1. Forward distance to nearest enemy (±1 if none in sight) -> 1 number
        # 2. Bearing to that enemy (sin θ, cos θ) -> 2 numbers
        # 3. Forward distance to nearest ally (or ±1) -> 1 number
        # 4. Bearing to that ally (sin θ, cos θ) -> 2 numbers
        # 5. Own health / 100 -> 1 number
        # 6. Weapon ready? (1 = yes, –1 = cooling) -> 1 number
        # 7. x-velocity, y-velocity (clamped) -> 2 numbers
        # 8. Bias neuron = 1 -> 1 number
        # TOTAL: 1+2+1+2+1+1+2+1 = 11 inputs.
        #
        # The TinyNet diagram shows `14 x 16`.
        # The provided `TinyNet` python code `np.random.uniform(-1, 1, (16, 14))` implies 14 inputs.
        # Let's assume the list is the source of truth for the *meaning* of inputs,
        # and if there are more inputs needed by TinyNet, they are just unused or need clarification.
        # For now, I will make the input vector 14 long and pad the rest with 0, except the bias.
        #
        # Re-evaluating the spec text description for inputs:
        # 1. Fwd dist enemy (1)
        # 2. Bearing enemy (sin, cos) (2)
        # 3. Fwd dist ally (1)
        # 4. Bearing ally (sin, cos) (2)
        # 5. Own health (1)
        # 6. Weapon ready (1)
        # 7. x-velocity, y-velocity (2)
        # 8. Bias neuron (1)
        # This sums to 11 inputs.
        # The TinyNet init `self.w_in = w_in if w_in is not None else np.random.uniform(-1, 1, (16, 14))`
        # clearly expects 14 inputs.
        # The example calculation `14 × 16 + 16 × 4 = 320 parameters` also uses 14.
        #
        # Possible missing inputs / interpretation:
        # - Own angle (sin, cos)? (2)
        # - Distance to nearest wall in N directions?
        # - Time since last fired?
        #
        # For now, I'll stick to the 11 defined inputs and pad the input vector to 14.
        # The last one must be the bias neuron.
        
        # Inputs indices used so far: 0-9
        # We need 14 inputs. Last one is bias.
        # inputs[10], inputs[11], inputs[12] are currently unused. Let's set them to 0.
        inputs[10] = 0.0 # Unused / Placeholder
        inputs[11] = 0.0 # Unused / Placeholder
        inputs[12] = 0.0 # Unused / Placeholder

        # Last input: Bias neuron = 1
        inputs[13] = 1.0
        
        return inputs


    def perform_actions_from_outputs(self, outputs, dt):
        """
        Applies actions based on the 4 NN outputs.
        Outputs are real values, thresholded at 0.
        """
        if not self.is_alive():
            self.vx, self.vy = 0,0
            self.current_speed = 0
            return

        # Outputs: o1, o2, o3, o4
        # o1: thrust forward / back
        # o2: strafe left / right
        # o3: rotate left / right
        # o4: fire / hold fire

        # --- Rotation (o3) ---
        rotation_input = outputs[2]
        if rotation_input >= 0: # Rotate left
            self.angle_deg -= self.rotation_speed_dps * dt
        else: # Rotate right
            self.angle_deg += self.rotation_speed_dps * dt
        self.angle_deg %= 360

        # --- Movement (o1: thrust, o2: strafe) ---
        thrust_input = outputs[0]
        strafe_input = outputs[1]
        
        target_vx, target_vy = 0.0, 0.0
        agent_angle_rad = math.radians(self.angle_deg)

        # Thrust
        if thrust_input >= 0: # Thrust forward
            target_vx += self.base_speed * math.cos(agent_angle_rad)
            target_vy += self.base_speed * math.sin(agent_angle_rad)
        else: # Thrust backward
            target_vx -= (self.base_speed / 2) * math.cos(agent_angle_rad) # Slower backward
            target_vy -= (self.base_speed / 2) * math.sin(agent_angle_rad)

        # Strafe
        strafe_speed_factor = 0.75 # Strafe a bit slower than forward thrust
        if strafe_input >= 0: # Strafe left
            # Left is agent_angle_rad - PI/2
            strafe_angle_rad = agent_angle_rad - math.pi / 2
            target_vx += self.base_speed * strafe_speed_factor * math.cos(strafe_angle_rad)
            target_vy += self.base_speed * strafe_speed_factor * math.sin(strafe_angle_rad)
        else: # Strafe right
            # Right is agent_angle_rad + PI/2
            strafe_angle_rad = agent_angle_rad + math.pi / 2
            target_vx += self.base_speed * strafe_speed_factor * math.cos(strafe_angle_rad)
            target_vy += self.base_speed * strafe_speed_factor * math.sin(strafe_angle_rad)
        
        # For "teleport-step", we directly set velocity for this frame
        self.vx = target_vx
        self.vy = target_vy
        
        # Update current_speed for consistency if other parts rely on it (e.g. bounce reduction)
        # This is a bit tricky as vx, vy are now composed. We can store the magnitude.
        self.current_speed = math.sqrt(self.vx**2 + self.vy**2)
        if (thrust_input < 0 or (strafe_input != 0 and thrust_input == 0)): # If primarily moving backward or only strafing
             # Heuristic: if dominant movement is backward or purely sideways, consider speed negative for some logic
             # This is not well-defined by vx,vy. Let's use thrust_input to determine "forward" intent for current_speed sign.
             if thrust_input < 0 and self.current_speed > 0: # Moving backward
                 self.current_speed *= -0.5 # Match backward thrust factor
             elif thrust_input == 0 and strafe_input != 0:
                 pass # Pure strafe, speed is positive magnitude
        elif thrust_input < 0: # Moving backward
            self.current_speed *= -1 # This might be an oversimplification

        # --- Firing (o4) ---
        fire_input = outputs[3]
        if fire_input >= 0: # Fire
            if self.weapon_cooldown_timer <= 0:
                self.is_firing_command = True
                self.weapon_cooldown_timer = self.weapon_cooldown_time
            else:
                self.is_firing_command = False # Can't fire, on cooldown
        else: # Hold fire
            self.is_firing_command = False


    def update(self, dt, arena_width=None, arena_height=None, all_agents=None):
        if not self.is_alive():
            return

        # Decrement weapon cooldown
        if self.weapon_cooldown_timer > 0:
            self.weapon_cooldown_timer -= dt
            if self.weapon_cooldown_timer < 0:
                self.weapon_cooldown_timer = 0
        
        # --- Decision Making ---
        if self.brain:
            if arena_width is None or arena_height is None or all_agents is None:
                # This check is important because get_inputs needs these.
                # For simple manual mode, these might not be passed.
                # print("Warning: Agent brain active but not all arena info provided to update.")
                pass # Or raise error, or skip brain update for this frame
            else:
                inputs = self.get_inputs(arena_width, arena_height, all_agents)
                outputs = self.brain(inputs)
                self.perform_actions_from_outputs(outputs, dt)
        elif not self.is_dummy: # Manual control
            # Rotation
            if self._manual_rotate_left: self.angle_deg -= self.rotation_speed_dps * dt
            if self._manual_rotate_right: self.angle_deg += self.rotation_speed_dps * dt
            self.angle_deg %= 360
            
            # Velocity based on thrust
            current_manual_speed = 0.0
            if self._manual_thrust_forward: current_manual_speed = self.base_speed
            elif self._manual_thrust_backward: current_manual_speed = -self.base_speed / 2
            
            angle_rad = math.radians(self.angle_deg)
            self.vx = current_manual_speed * math.cos(angle_rad)
            self.vy = current_manual_speed * math.sin(angle_rad)
            self.current_speed = current_manual_speed # Store the signed speed

            # Firing for manual
            if self._manual_fire and self.weapon_cooldown_timer <= 0:
                self.is_firing_command = True
                self.weapon_cooldown_timer = self.weapon_cooldown_time
            else:
                self.is_firing_command = False # Reset if not firing or on cooldown
        else: # Dummy
            self.vx, self.vy = 0,0
            self.current_speed = 0
            self.is_firing_command = False

        # --- Position Update (Teleport-step) ---
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Reset manual states (they are read fresh from keys each frame by viewer)
        # self._manual_thrust_forward = False ... (not needed here)

    def draw(self, screen):
        if not self.is_alive():
            # Optionally draw a wreck or nothing
            pygame.draw.circle(screen, (50,50,50), (int(self.x), int(self.y)), self.radius, 2) # Grey outline
            return

        # Draw body (circle)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw direction line
        angle_rad = math.radians(self.angle_deg)
        end_x = self.x + self.radius * math.cos(angle_rad)
        end_y = self.y + self.radius * math.sin(angle_rad)
        pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)

        # Draw HP bar (optional)
        if self.max_hp > 0:
            hp_bar_width = self.radius * 2
            hp_bar_height = 5
            hp_bar_x = self.x - self.radius
            hp_bar_y = self.y - self.radius - hp_bar_height - 2 # Above agent

            current_hp_width = (self.hp / self.max_hp) * hp_bar_width
            pygame.draw.rect(screen, (255,0,0), (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height)) # Red background
            pygame.draw.rect(screen, (0,255,0), (hp_bar_x, hp_bar_y, current_hp_width, hp_bar_height)) # Green foreground
        
        # Draw weapon cooldown indicator (optional)
        if self.weapon_cooldown_timer > 0:
            cooldown_arc_angle = (self.weapon_cooldown_timer / self.weapon_cooldown_time) * 360
            if cooldown_arc_angle > 0:
                try:
                    # Simplified: draw a small arc or circle segment
                    rect = pygame.Rect(self.x - self.radius/2, self.y - self.radius/2, self.radius, self.radius)
                    pygame.draw.arc(screen, (200,200,0), rect, 0, math.radians(cooldown_arc_angle), 2) # Yellow
                except TypeError as e: # pygame.draw.arc can be picky with small angles
                    # print(f"Warning: Pygame arc drawing issue: {e}")
                    pass


    def get_state_for_replay(self):
        """Returns a serializable dict of the agent's current state for replay."""
        return {
            'id': self.agent_id,
            'team_id': self.team_id,
            'x': round(self.x, 2),
            'y': round(self.y, 2),
            'angle_deg': round(self.angle_deg, 2),
            'hp': round(self.hp, 1),
            'is_firing': self.is_firing_command, # Log if trying to fire this frame
            'vx': round(self.vx, 2),
            'vy': round(self.vy, 2)
        }