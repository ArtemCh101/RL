import os
import sys

import pandas as pd
import numpy as np
from collections import defaultdict
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from custom_agents import CustomQLAgent
from generator import TrafficGenerator

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    runs = 30
    episodes = 4

    simulation_seconds = 10000

    NET_FILE = "nets/4x4-Lucas/4x4.net.xml"
    ROUTE_FILE = "nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"

    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,
        num_seconds=simulation_seconds,
        min_green=5,
        delta_time=5,
        single_agent=False
    )

    all_runs_final_results = []

    for run in range(1, runs + 1):
        initial_states = env.reset()

        ql_agents = {}
        for ts in env.ts_ids:
            ql_agents[ts] = CustomQLAgent(
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                epsilon=0.05,
                min_epsilon=0.005,
                epsilon_decay=1.0
            )

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()

            done = {"__all__": False}
            episode_reward = 0

            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act(initial_states[ts]) for ts in ql_agents.keys()}

                s, r, done_combined, info = env.step(action=actions)

                if env.sim_step % 2500 == 0:
                    print(
                        f"QL Run {run}, Episode {episode}/{episodes}: Current Sim Time {env.sim_step}/{simulation_seconds}")

                done = done_combined

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(
                        state=initial_states[agent_id],
                        action=actions[agent_id],
                        reward=r[agent_id],
                        next_state=s[agent_id],
                        done=done["__all__"]
                    )

                initial_states = s
                episode_reward += sum(r.values())

            if episode == episodes:
                final_episode_data = {
                    'run': run,
                    'episode': episode,
                    'reward': episode_reward,
                    'algorithm': 'Custom_QLearning_4x4',
                    'epsilon': ql_agents[env.ts_ids[0]].epsilon
                }
                all_runs_final_results.append(final_episode_data)

            print(f"QL Run {run}, Episode {episode}/{episodes}: Reward {episode_reward:.2f}")

    df_final = pd.DataFrame(all_runs_final_results)
    df_final.to_csv("../outputs/ql_final_episodes_summary.csv", index=False)
    print("Q-Learning finished. Final episode data saved to ../outputs/ql_final_episodes_summary.csv")

    env.close()