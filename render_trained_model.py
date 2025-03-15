import os
import imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from src import envs
from src.maddpg import MADDPG
from src.utils import transpose_to_tensor

def create_frame(world, width=400, height=400):
    """Create a frame using PIL instead of matplotlib"""
    # Create a new image with a white background
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Convert world coordinates to image coordinates
    def world_to_image(x, y):
        # Convert from [-1.5, 1.5] to [0, width/height]
        ix = int((x + 1.5) * width / 3)
        iy = int((1.5 - y) * height / 3)  # Flip y-axis
        return ix, iy
    
    # Draw grid lines
    for i in range(4):
        x = -1.5 + i * 1.0
        start_x, start_y = world_to_image(x, -1.5)
        end_x, end_y = world_to_image(x, 1.5)
        draw.line([(start_x, start_y), (end_x, end_y)], fill='lightgray', width=1)
        
        start_x, start_y = world_to_image(-1.5, x)
        end_x, end_y = world_to_image(1.5, x)
        draw.line([(start_x, start_y), (end_x, end_y)], fill='lightgray', width=1)
    
    # Draw entities
    for entity in world.entities:
        # Convert color from [0,1] to [0,255]
        color = tuple(int(c * 255) for c in entity.color[:3])
        position = entity.state.p_pos
        size = int(entity.size * width / 3)  # Scale the size appropriately
        
        # Convert position to image coordinates
        x, y = world_to_image(position[0], position[1])
        
        # Draw circle
        draw.ellipse([x-size, y-size, x+size, y+size], 
                     fill=color + (128,) if 'agent' in entity.name else color,
                     outline='black')
        
        # Add label for agents
        if 'agent' in entity.name:
            # Calculate text size and position
            label_size = draw.textlength(entity.name)
            text_x = x - label_size/2
            text_y = y - 5
            draw.text((text_x, text_y), entity.name, fill='black')
    
    # Convert PIL image to numpy array
    return np.array(image)

def render_episode(model_path, episode_num=5, save_gif=True):
    # Initialize environment with single environment
    env = envs.make_env("simple_adversary")
    
    # Initialize MADDPG and load trained model
    maddpg = MADDPG()
    
    # Load the saved model
    save_dict_list = torch.load(model_path, weights_only=True)
    for i in range(3):
        maddpg.maddpg_agent[i].actor.load_state_dict(save_dict_list[i]['actor_params'])
        maddpg.maddpg_agent[i].actor_optimizer.load_state_dict(save_dict_list[i]['actor_optim_params'])
        maddpg.maddpg_agent[i].critic.load_state_dict(save_dict_list[i]['critic_params'])
        maddpg.maddpg_agent[i].critic_optimizer.load_state_dict(save_dict_list[i]['critic_optim_params'])
    
    print(f"Loaded model from {model_path}")
    
    for ep in range(episode_num):
        print(f"\nRendering episode {ep+1}/{episode_num}")
        obs_n, obs_full = env.reset()
        frames = []
        
        # Run one episode
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 200:
            # Get actions for all agents
            actions = maddpg.act(transpose_to_tensor([obs_n]), noise=0.0)
            actions_array = torch.stack(actions).detach().numpy()
            actions_for_env = np.squeeze(actions_array, axis=1)
            
            # Environment step
            obs_n, obs_full, rewards, dones, info = env.step(actions_for_env)
            episode_reward += np.mean(rewards)
            
            # Create and save frame
            frame = create_frame(env.world)
            frames.append(frame)
            
            done = any(dones)
            step += 1
        
        print(f"Episode {ep+1} finished with reward: {episode_reward:.2f}")
        
        # Save gif
        if save_gif and frames:
            gif_path = os.path.join(os.path.dirname(model_path), f'visualization_ep_{ep+1}.gif')
            imageio.mimsave(gif_path, frames, duration=0.1)
            print(f"Saved gif to {gif_path}")
    
    env.close()

def main():
    # Get the latest model file from model_dir
    model_dir = os.path.join(os.getcwd(), "model_dir")
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not model_files:
        print("No model files found in model_dir!")
        return
    
    # Sort by episode number
    latest_model = sorted(model_files, key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    print(f"Using model: {latest_model}")
    
    # Render episodes
    render_episode(model_path, episode_num=5, save_gif=True)

if __name__ == "__main__":
    main()