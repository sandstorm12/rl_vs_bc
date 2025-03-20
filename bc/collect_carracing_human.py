import os
import cv2
import yaml
import shelve
import argparse
import gymnasium as gym
from tqdm import tqdm


ACTION_MAP = {
    ord('w'): 3,  # Accelerate
    ord('s'): 4,  # Brake
    ord('a'): 2,  # Left
    ord('d'): 1,  # Right
    -1: 0        # No action (when no key pressed)
}

def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)
    
    return configs


def _load_env():
    env = gym.make("CarRacing-v2", continuous=False, render_mode='human')
    
    return env


def _collect_human_data(configs):
    if os.path.exists(configs['store_path']) and not configs['overwrite']:
        print("Data already exists and overwrite is disabled. Exiting.")
        return
    
    env = _load_env()
    
    with shelve.open(configs['store_path']) as storage:
        obs, _ = env.reset()
        episode_length = 0
        idx_sample = 0

        cv2.namedWindow("Observation", cv2.WINDOW_NORMAL)
        
        while idx_sample < configs['num_samples']:
            key = cv2.waitKey(40) & 0xFF  # 33ms delay ≈ 30 FPS
            action = ACTION_MAP.get(key, 0)  # Default to no action
            
            if key == ord('q'):
                print("Quitting data collection.")
                break

            obs_new, reward, done, truncated, info = env.step(action)
            
            obs_display = cv2.resize(obs, (512, 512))
            cv2.imshow("Observation", obs_display)

            storage[str(idx_sample)] = (
                obs, obs_new, action,
                reward, done, truncated, info
            )

            if idx_sample % 100 == 0:
                print(f"Collected {idx_sample}/{configs['num_samples']} samples")

            if done or truncated:
                obs_new, _ = env.reset()
                episode_length = 0
                print("Episode reset")
            
            obs = obs_new
            episode_length += 1
            idx_sample += 1
            
        env.close()
        cv2.destroyAllWindows()
        print(f"Collected {idx_sample} human demonstration samples")

if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)
    
    print(f"Config loaded: {configs}")
    _collect_human_data(configs)
