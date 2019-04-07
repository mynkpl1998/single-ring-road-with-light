from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
sys.path.append(os.getcwd() + "/")

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

import yaml
import os, sys
sys.path.append(os.getcwd() + "/")

from SingleLaneIDMAge.SimulatorCode.main_env import Wrapper
from SingleLaneIDMAge.SimulatorCode.controllers import PPORLController
from SingleLaneIDMAge.Independent.TrainingScripts.ppo_train import policy_mapper, gen_policy
from ray.rllib.agents.ppo.ppo import PPOAgent

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--episode-length", type=int, default=3000, help="max length of the episode")
parser.add_argument("--num-episodes", type=int, default=100, help="number of epsiodes")
parser.add_argument("--density", default=0.7, type=float, help="vehicle density")
parser.add_argument("--checkpoint-file", type=str, help="checkpoint file path")
parser.add_argument("--render", default=1, type=int, help="renders simulation screen")
parser.add_argument("--sim-config-file", type=str, help="simulation configuration file")
parser.add_argument("--ppo-config-file", type=str, help="ppo experiment configuration file")


if __name__ == "__main__":

	args = parser.parse_args()
	
	trajec_path = "/SingleLaneIDM/SimulatorCode/micro.pkl"

	with open(args.sim_config_file, "r") as handle:
		sim_config = yaml.load(handle)

	with open(args.ppo_config_file, "r") as handle:
		exp_config = yaml.load(handle)

	if args.render == 1:
		sim_config["config"]["render"] = True
	else:
		sim_config["config"]["render"] = False

	sim_config["config"]["trajec-file-path"] = os.getcwd() + trajec_path
	sim_config["config"]["external-controller"] = True

	exp_name = list(exp_config.keys())[0]
	exp_config[exp_name]["config"]["horizon"] = int(sim_config["config"]["horizon"])
	exp_config[exp_name]["config"]["callbacks"]["on_episode_end"] = None

	env_creator_name = "tsim-v0"
	register_env(env_creator_name, lambda config: Wrapper(sim_config))
		
	env = Wrapper(sim_config)
	
	policy_graphs = {
		"comm_policy": gen_policy("comm", exp_config[exp_name]["config"], env),
		"planner_policy": gen_policy("planner", exp_config[exp_name]["config"], env)
	}

	exp_config[exp_name]["config"]["multiagent"]["policy_graphs"] = policy_graphs
	exp_config[exp_name]["config"]["multiagent"]["policy_mapping_fn"] = tune.function(policy_mapper)

	#controller = PPORLController(False, sim_config, exp_config, args.checkpoint_file)

	ray.init()
	

	agent = PPOAgent(env="tsim-v0", config={
		"observation_filter": "NoFilter",
		"multiagent": {
			"policy_mapping_fn": policy_mapper,
			"policy_graphs": {
				"comm_policy": (PPOPolicyGraph, env.comm_observation_space, env.comm_action_space, {}),
				"planner_policy": (PPOPolicyGraph, env.planner_observation_space, env.planner_action_space, {})
			},
			"policies_to_train": ["comm_policy", "planner_policy"]
		}
		})

	agent.restore(args.checkpoint_file)
	

	episodes = args.num_episodes
	horizon = args.episode_length

	
	for episode in range(0, episodes):

		prev_state = env.reset(args.density)
		episode_reward = 0.0

		for step in range(0, horizon):

			action = {}
			action["planner"] = agent.compute_action(prev_state["planner"], policy_id="planner_policy")
			action["comm"] = agent.compute_action(prev_state["comm"], policy_id="comm_policy")

			next_state, reward, done, done_dict = env.step(action)

			episode_reward += reward["planner"]
			prev_state = next_state

			if done["__all__"]:
				break

		print("Episode Lasted for %d time steps and accumulated %.2f Reward"%(step+1, episode_reward))
	