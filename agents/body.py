# evo_arena/agents/body.py
import pygame
import math

class AgentBody:
    def __init__(self, x, y, angle_deg, base_speed, rotation_speed, radius=10, color=(0, 0, 255), is_dummy=False, agent_id="agent"):
        self.agent_id = agent_id
        self.x = float(x)
        self.y = float(y)
        self.angle_deg = float(angle_deg)  # Degrees
        self.radius = int(radius)
        self.color = color
        self.is_dummy = is_dummy

        self.base_speed = float(base_speed) # "meters" / second
        self.rotation_speed_dps = float(rotation_speed) # degrees / second
        
        self.vx = 0.0  # Velocity in x direction
        self.vy = 0.0  # Velocity in y direction
        self.current_speed = 0.0 # aktuellen Vorwärts/Rückwärts Speed
        
        # For manual control state
        self.is_thrusting_forward = False
        self.is_thrusting_backward = False
        self.is_rotating_left = False
        self.is_rotating_right = False

    def manual_control(self, keys):
        if self.is_dummy:
            return

        self.is_thrusting_forward = keys[pygame.K_UP]
        self.is_thrusting_backward = keys[pygame.K_DOWN]
        self.is_rotating_left = keys[pygame.K_LEFT]
        self.is_rotating_right = keys[pygame.K_RIGHT]

    def update(self, dt):
        if self.is_dummy:
            # Dummy agent does not move or rotate unless explicitly set
            return

        # --- Rotation ---
        if self.is_rotating_left:
            self.angle_deg -= self.rotation_speed_dps * dt
        if self.is_rotating_right:
            self.angle_deg += self.rotation_speed_dps * dt
        self.angle_deg %= 360 # Keep angle within 0-360

        # --- Velocity based on thrust ---
        self.current_speed = 0.0
        if self.is_thrusting_forward:
            self.current_speed = self.base_speed
        elif self.is_thrusting_backward:
            self.current_speed = -self.base_speed / 2 # Move slower backwards

        angle_rad = math.radians(self.angle_deg)
        self.vx = self.current_speed * math.cos(angle_rad)
        self.vy = self.current_speed * math.sin(angle_rad)
        
        # --- Position Update (Teleport-step) ---
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Reset thrust/rotation states for next frame if you want press-to-activate
        # If not reset, they remain active as long as key is held (handled by manual_control re-evaluating keys)
        # self.is_thrusting_forward = False 
        # ... (this is better handled by re-reading keys each frame)

    def draw(self, screen):
        # Draw body (circle)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw direction line
        angle_rad = math.radians(self.angle_deg)
        end_x = self.x + self.radius * math.cos(angle_rad)
        end_y = self.y + self.radius * math.sin(angle_rad)
        pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)