# evo_arena/agents/body.py
import pygame
import math
import numpy as np
import random # For the jitter

class AgentBody:
    def __init__(self, x, y, angle_deg, base_speed, rotation_speed_dps,
                 radius=15, color=(0, 0, 255), agent_id="agent", team_id=0,
                 hp=100, brain=None, is_dummy=False,
                 weapon_range=80, weapon_arc_deg=120, weapon_cooldown_time=0.6, weapon_damage=10,
                 cooldown_jitter_factor=0.1): # New parameter for jitter, e.g., 0.1 for +/-10%
        
        self.agent_id = str(agent_id)
        self.team_id = int(team_id)
        self.x = float(x)
        self.y = float(y)
        self.angle_deg = float(angle_deg) 
        self.radius = int(radius)
        self.color = color
        self.is_dummy = is_dummy
        self.brain = brain 

        self.base_speed = float(base_speed) 
        self.rotation_speed_dps = float(rotation_speed_dps) 
        
        self.vx = 0.0 
        self.vy = 0.0 
        
        self.max_hp = float(hp)
        self.hp = float(hp)
        
        self.weapon_range = float(weapon_range)
        self.weapon_arc_deg = float(weapon_arc_deg) 
        self.base_weapon_cooldown_time = float(weapon_cooldown_time) # Store the base cooldown
        self.weapon_damage = int(weapon_damage)
        self.weapon_cooldown_timer = 0.0 
        self.is_firing_command = False 

        self._manual_thrust_forward = False
        self._manual_thrust_backward = False
        self._manual_rotate_left = False
        self._manual_rotate_right = False
        self._manual_fire = False
        
        self.max_abs_velocity_component = self.base_speed 
        self.relative_angle_to_enemy_rad_for_input = 0.0

        self.cooldown_jitter_factor = float(cooldown_jitter_factor) # Store jitter factor

        # For reproducibility of jitter per agent instance if desired,
        # but usually global random is fine for this kind of noise.
        # self.rng_jitter = random.Random() 
        # self.rng_jitter.seed(hash(self.agent_id)) # Seed with agent_id for consistent jitter *for this agent* if re-created identically

    def get_effective_cooldown_time(self):
        """Calculates the actual cooldown time for the next shot, including jitter."""
        if self.cooldown_jitter_factor == 0:
            return self.base_weapon_cooldown_time
        
        jitter = random.uniform(-self.base_weapon_cooldown_time * self.cooldown_jitter_factor,
                                 self.base_weapon_cooldown_time * self.cooldown_jitter_factor)
        effective_cooldown = self.base_weapon_cooldown_time + jitter
        return max(0.05, effective_cooldown) # Ensure cooldown doesn't become zero or negative, minimum 50ms

    def is_alive(self):
        return self.hp > 0

    def take_damage(self, amount):
        if not self.is_alive(): return
        self.hp -= amount
        if self.hp < 0:
            self.hp = 0
        if self.hp == 0:
            print(f"Agent {self.agent_id} (Team {self.team_id}) defeated.")

    def manual_control(self, keys):
        if self.is_dummy or self.brain is not None:
            return

        self._manual_thrust_forward = keys[pygame.K_UP]
        self._manual_thrust_backward = keys[pygame.K_DOWN]
        self._manual_rotate_left = keys[pygame.K_LEFT]
        self._manual_rotate_right = keys[pygame.K_RIGHT]
        self._manual_fire = keys[pygame.K_SPACE]

    def get_inputs(self, arena_width, arena_height, all_agents):
        inputs = np.zeros(self.brain.input_size if self.brain else 14)

        def normalize(value, min_val, max_val):
            if max_val == min_val: return 0.0 
            clamped_value = np.clip(value, min_val, max_val)
            return 2 * (clamped_value - min_val) / (max_val - min_val) - 1
        
        def find_nearest_agent_in_list(agent_list):
            nearest_dist = float('inf')
            nearest_agent_obj = None
            for other_agent in agent_list:
                if other_agent is self or not other_agent.is_alive():
                    continue
                dx = other_agent.x - self.x
                dy = other_agent.y - self.y
                dist = math.sqrt(dx**2 + dy**2) - self.radius - other_agent.radius
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_agent_obj = other_agent
            return nearest_agent_obj, nearest_dist

        max_view_dist = math.sqrt(arena_width**2 + arena_height**2)
        agent_angle_rad = math.radians(self.angle_deg)

        enemies = [a for a in all_agents if a.team_id != self.team_id and a.is_alive()]
        nearest_enemy, enemy_dist = find_nearest_agent_in_list(enemies)
        
        aim_quality_cos_to_enemy = -1.0 

        if nearest_enemy:
            inputs[0] = normalize(enemy_dist, 0, max_view_dist)
            dx_enemy = nearest_enemy.x - self.x
            dy_enemy = nearest_enemy.y - self.y
            angle_to_enemy_rad = math.atan2(dy_enemy, dx_enemy)
            
            self.relative_angle_to_enemy_rad_for_input = (angle_to_enemy_rad - agent_angle_rad + math.pi) % (2 * math.pi) - math.pi
            
            inputs[1] = math.sin(self.relative_angle_to_enemy_rad_for_input) 
            inputs[2] = math.cos(self.relative_angle_to_enemy_rad_for_input) 
            aim_quality_cos_to_enemy = math.cos(self.relative_angle_to_enemy_rad_for_input)
        else:
            inputs[0] = 1.0 
            inputs[1] = 0.0 
            inputs[2] = 0.0
            self.relative_angle_to_enemy_rad_for_input = 0.0 

        allies = [a for a in all_agents if a.team_id == self.team_id and a is not self and a.is_alive()]
        nearest_ally, ally_dist = find_nearest_agent_in_list(allies)
        if nearest_ally:
            inputs[3] = normalize(ally_dist, 0, max_view_dist)
            dx_ally = nearest_ally.x - self.x
            dy_ally = nearest_ally.y - self.y
            angle_to_ally_rad = math.atan2(dy_ally, dx_ally)
            relative_angle_ally_rad = (angle_to_ally_rad - agent_angle_rad + math.pi) % (2 * math.pi) - math.pi
            inputs[4] = math.sin(relative_angle_ally_rad)
            inputs[5] = math.cos(relative_angle_ally_rad)
        else:
            inputs[3] = 1.0 
            inputs[4] = 0.0
            inputs[5] = 0.0

        inputs[6] = normalize(self.hp, 0, self.max_hp)
        inputs[7] = 1.0 if self.weapon_cooldown_timer <= 0 else -1.0
        inputs[8] = normalize(self.vx, -self.max_abs_velocity_component, self.max_abs_velocity_component)
        inputs[9] = normalize(self.vy, -self.max_abs_velocity_component, self.max_abs_velocity_component)
        inputs[10] = aim_quality_cos_to_enemy
        inputs[11] = 0.0 
        inputs[12] = 0.0 

        if inputs.shape[0] == 14: 
            inputs[13] = 1.0
        
        return inputs

    def perform_actions_from_outputs(self, outputs, dt):
        if not self.is_alive():
            self.vx, self.vy = 0, 0
            return

        rotation_input = outputs[2]
        if rotation_input >= 0: 
            self.angle_deg -= self.rotation_speed_dps * dt
        else: 
            self.angle_deg += self.rotation_speed_dps * dt
        self.angle_deg %= 360 

        thrust_input = outputs[0]  
        strafe_input = outputs[1]  
        
        target_vx, target_vy = 0.0, 0.0
        agent_angle_rad = math.radians(self.angle_deg)

        current_thrust_speed = 0.0
        if thrust_input >= 0: 
            current_thrust_speed = self.base_speed
        else: 
            current_thrust_speed = -self.base_speed * 0.5 
        
        target_vx += current_thrust_speed * math.cos(agent_angle_rad)
        target_vy += current_thrust_speed * math.sin(agent_angle_rad)

        current_strafe_speed = 0.0
        strafe_speed_factor = 0.75 
        if strafe_input != 0: 
            strafe_angle_rad = 0 
            if strafe_input >= 0: 
                strafe_angle_rad = agent_angle_rad - math.pi / 2
                current_strafe_speed = self.base_speed * strafe_speed_factor
            else: 
                strafe_angle_rad = agent_angle_rad + math.pi / 2
                current_strafe_speed = self.base_speed * strafe_speed_factor
            
            target_vx += current_strafe_speed * math.cos(strafe_angle_rad)
            target_vy += current_strafe_speed * math.sin(strafe_angle_rad)
        
        self.vx = target_vx
        self.vy = target_vy
        
        current_speed_sq = self.vx**2 + self.vy**2
        if current_speed_sq > self.base_speed**2 and self.base_speed > 0:
            scale = self.base_speed / math.sqrt(current_speed_sq)
            self.vx *= scale
            self.vy *= scale

        fire_input = outputs[3]
        if fire_input >= 0: 
            if self.weapon_cooldown_timer <= 0:
                self.is_firing_command = True
                self.weapon_cooldown_timer = self.get_effective_cooldown_time() # MODIFIED: Use jittered cooldown
            else:
                self.is_firing_command = False 
        else: 
            self.is_firing_command = False

    def update(self, dt, arena_width=None, arena_height=None, all_agents=None):
        if not self.is_alive():
            self.vx, self.vy = 0,0 
            return

        if self.weapon_cooldown_timer > 0:
            self.weapon_cooldown_timer -= dt
            if self.weapon_cooldown_timer < 0:
                self.weapon_cooldown_timer = 0
        
        if self.brain: 
            if arena_width is None or arena_height is None or all_agents is None:
                self.vx, self.vy = 0, 0
                self.is_firing_command = False
            else:
                inputs = self.get_inputs(arena_width, arena_height, all_agents)
                outputs = self.brain(inputs)
                self.perform_actions_from_outputs(outputs, dt)
        elif not self.is_dummy: 
            if self._manual_rotate_left: self.angle_deg -= self.rotation_speed_dps * dt
            if self._manual_rotate_right: self.angle_deg += self.rotation_speed_dps * dt
            self.angle_deg %= 360
            
            manual_target_speed = 0.0
            if self._manual_thrust_forward: manual_target_speed = self.base_speed
            elif self._manual_thrust_backward: manual_target_speed = -self.base_speed * 0.5
            
            angle_rad = math.radians(self.angle_deg)
            self.vx = manual_target_speed * math.cos(angle_rad)
            self.vy = manual_target_speed * math.sin(angle_rad)

            if self._manual_fire:
                if self.weapon_cooldown_timer <= 0:
                    self.is_firing_command = True
                    self.weapon_cooldown_timer = self.get_effective_cooldown_time() # MODIFIED: Use jittered cooldown
                else:
                    self.is_firing_command = False 
            else: 
                self.is_firing_command = False
        else: 
            self.vx, self.vy = 0,0
            self.is_firing_command = False

        self.x += self.vx * dt
        self.y += self.vy * dt
        
    def draw(self, screen):
        body_color = self.color
        if not self.is_alive():
            body_color = (50, 50, 50) 
        pygame.draw.circle(screen, body_color, (int(self.x), int(self.y)), self.radius)
        
        if not self.is_alive(): 
            return

        angle_rad = math.radians(self.angle_deg)
        end_x = self.x + self.radius * math.cos(angle_rad)
        end_y = self.y + self.radius * math.sin(angle_rad)
        pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)

        if self.max_hp > 0:
            hp_bar_width = self.radius * 1.5 
            hp_bar_height = 5
            hp_bar_x = self.x - hp_bar_width / 2
            hp_bar_y = self.y - self.radius - hp_bar_height - 3 

            current_hp_ratio = self.hp / self.max_hp
            current_hp_width = current_hp_ratio * hp_bar_width
            
            pygame.draw.rect(screen, (150,0,0), (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
            if current_hp_width > 0:
                pygame.draw.rect(screen, (0,200,0), (hp_bar_x, hp_bar_y, current_hp_width, hp_bar_height))
        
        if self.weapon_cooldown_timer > 0 and self.base_weapon_cooldown_time > 0: # Check base_cooldown_time to avoid div by zero if it's 0
            # For display, normalize against the base cooldown time, not the potentially jittered one
            cooldown_ratio = self.weapon_cooldown_timer / self.base_weapon_cooldown_time 
            cooldown_ratio = min(1.0, cooldown_ratio) # Cap at 1.0 for display if jitter made it slightly longer

            arc_radius = self.radius * 0.6
            arc_rect = pygame.Rect(self.x - arc_radius, self.y - arc_radius, arc_radius*2, arc_radius*2)
            start_angle = -math.pi/2 
            end_angle = start_angle + (2 * math.pi * cooldown_ratio)
            try:
                 pygame.draw.arc(screen, (200,200,0), arc_rect, start_angle, end_angle, 2) 
            except TypeError: 
                 pass

    def get_state_for_replay(self):
        return {
            'id': self.agent_id,
            'team_id': self.team_id,
            'x': round(self.x, 2),
            'y': round(self.y, 2),
            'angle_deg': round(self.angle_deg, 2),
            'hp': round(self.hp, 1),
            'is_alive': self.is_alive(),
            'is_firing_cmd': self.is_firing_command, 
            'vx': round(self.vx, 2), 
            'vy': round(self.vy, 2)
        }

    def reset_state(self, x, y, angle_deg, hp=None):
        self.x = float(x)
        self.y = float(y)
        self.angle_deg = float(angle_deg)
        self.hp = float(hp if hp is not None else self.max_hp)
        self.vx = 0.0
        self.vy = 0.0
        self.weapon_cooldown_timer = 0.0 # Initial cooldown can be 0 after reset, jitter applies on first fire
        self.is_firing_command = False
        self._manual_thrust_forward = False
        self._manual_thrust_backward = False
        self._manual_rotate_left = False
        self._manual_rotate_right = False
        self._manual_fire = False
        self.relative_angle_to_enemy_rad_for_input = 0.0