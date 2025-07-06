import pygame
import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

WIDTH, HEIGHT = 600, 400
CART_WIDTH, CART_HEIGHT = 80, 20
POLE_LENGTH = 100

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 24), nn.ReLU(),
            nn.Linear(24, 48), nn.ReLU(),
            nn.Linear(48, action_size)
        )
    def forward(self, x): return self.layers(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state_t = torch.FloatTensor(state).to(self.device)
            next_state_t = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_t)).item()
            output = self.model(state_t)
            target_f = output.clone().detach()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.loss_fn(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Pygame drawing functions
def draw_cartpole(screen, state):
    x, _, theta, _ = state
    # Convert state to screen coordinates
    cart_x = WIDTH // 2 + int(x * 100)  # scale position
    cart_y = HEIGHT // 2 + 50

    # Draw cart (rectangle)
    pygame.draw.rect(screen, (0, 0, 255), (cart_x - CART_WIDTH // 2, cart_y, CART_WIDTH, CART_HEIGHT))

    # Calculate pole end point
    pole_x = cart_x + POLE_LENGTH * math.sin(theta)
    pole_y = cart_y - POLE_LENGTH * math.cos(theta)

    # Draw pole (line)
    pygame.draw.line(screen, (255, 0, 0), (cart_x, cart_y), (pole_x, pole_y), 5)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CartPole RL with GUI")

    clock = pygame.time.Clock()
    env = gym.make("CartPole-v1", render_mode=None)  # no gym rendering, we'll do our own
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    episodes = 500
    font = pygame.font.SysFont("Arial", 20)

    for e in range(episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

            # Draw everything
            screen.fill((255, 255, 255))
            draw_cartpole(screen, state)
            text = font.render(f"Episode: {e+1}  Reward: {int(total_reward)}  Epsilon: {agent.epsilon:.2f}", True, (0, 0, 0))
            screen.blit(text, (10, 10))

            pygame.display.flip()
            clock.tick(60)  # 60 FPS

        print(f"Episode {e+1}/{episodes} finished with reward: {total_reward}")

    pygame.quit()

if __name__ == "__main__":
    main()
