import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

sys.path.append(PROJECT_ROOT)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from custom_agents import CustomDQNAgent

RUNS = 10
EPISODES = 10
SIMULATION_SECONDS = 5400
ALGORITHM_NAME = "Custom_DQN_PER_BigIntersection"

NET_FILE = os.path.join(PROJECT_ROOT, "nets", "big-intersection", "big-intersection.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "big-intersection", "routes.rou.xml")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "big-intersection")
OUT_CSV_NAME = os.path.join(OUT_DIR, "dqn_per")

AGENT_HYPERPARAMS = {
    "hidden_dims": (256, 256),
    "lr": 1e-4,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.9999,
    "min_epsilon": 0.05,
    "buffer_size": 20000,
    "batch_size": 64,
    "target_update_steps": 500,
    "train_freq": 1
}

os.makedirs(OUT_DIR, exist_ok=True)


def run_experiment():
    env = SumoEnvironment(
        net_file=NET_FILE,
        single_agent=True,
        route_file=ROUTE_FILE,
        out_csv_name=OUT_CSV_NAME,
        use_gui=False,
        num_seconds=SIMULATION_SECONDS,
        yellow_time=4,
        min_green=5,
        max_green=60,
        delta_time=5
    )

    all_runs_final_results = []

    state_dim = env.observation_space.shape[0]

    initial_state, _ = env.reset()
    ts_id = env.ts_ids[0]

    for run in range(1, RUNS + 1):
        print(f"\n--- Starting Run {run}/{RUNS} ---")

        if run != 1:
            initial_state, _ = env.reset()

        agent = CustomDQNAgent(
            action_space=env.action_space,
            state_dim=state_dim,
            **AGENT_HYPERPARAMS
        )
        agent.epsilon = AGENT_HYPERPARAMS["epsilon"]

        for episode in range(1, EPISODES + 1):
            if episode != 1:
                initial_state, _ = env.reset()

            current_state = initial_state
            done = False
            episode_reward = 0

            while not done:
                action = agent.act(current_state)

                next_state, reward, terminated, truncated, info = env.step(action=action)

                done = terminated or truncated

                agent.learn(current_state, action, reward, next_state, done)

                current_state = next_state
                episode_reward += reward

                if env.sim_step % 5000 == 0:
                    print(
                        f"Run {run}, Episode {episode}: Sim Time {env.sim_step}/{SIMULATION_SECONDS}, Epsilon: {agent.epsilon:.4f}, Q-Steps: {agent.current_step}")

            print(f"Run {run}, Episode {episode}/{EPISODES}: Reward {episode_reward:.2f}")

            if episode == EPISODES:
                final_episode_data = {
                    'run': run,
                    'episode': episode,
                    'total_reward': episode_reward,
                    'algorithm': ALGORITHM_NAME,
                    'final_epsilon': agent.epsilon,
                    'total_steps': agent.current_step,
                }
                all_runs_final_results.append(final_episode_data)

    env.close()

    df_final = pd.DataFrame(all_runs_final_results)

    final_summary_path = os.path.join(OUT_DIR, f"{ALGORITHM_NAME}_final_episodes_summary.csv")
    df_final.to_csv(final_summary_path, index=False)

    print("\n" + "=" * 70)
    print(f"DQN-PER experiment finished ({RUNS} runs, {EPISODES} episodes each).")
    print(f"Final episode summary saved to {final_summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_experiment()