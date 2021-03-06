import os, sys
sys.path.append(os.getcwd() + "/")

from SingleLaneIDM.SimulatorCode.main_env import Wrapper 
from SingleLaneIDM.SimulatorCode.main_env import ManualController, ApexRLController, PPORLController
import argparse
import numpy as np
import pickle
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--episode-length", type=int, default=3000, help="max length of the episode")
parser.add_argument("--num-episodes", type=int, default=10, help="number of epsiodes")
parser.add_argument("--checkpoint-file", type=str, help="checkpoint file path")
parser.add_argument("--sim-config-file", type=str, help="simulation configuration file")
parser.add_argument("--ppo-config-file", type=str, help="ppo experiment configuration file")
parser.add_argument("--save-file-name-path", type=str, help="file name for saving simulated data")
parser.add_argument("--allow-pertubations", type=int, default=0, help="whether to allow traffic light pertubations or not")

if __name__ == "__main__":

	args = parser.parse_args()

	trajec_path = "/SingleLaneIDM/SimulatorCode/micro.pkl"
	test_trajec_path = "/SingleLaneIDM//SimulatorCode/tf_time_steps.pkl"

	global_data_dict = {}
	densities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
	
	with open(args.sim_config_file, "r") as handle:
		sim_config = yaml.load(handle)

	with open(args.ppo_config_file, "r") as handle:
		exp_config = yaml.load(handle)

	sim_config["config"]["render"] = False

	sim_config["config"]["trajec-file-path"] = os.getcwd() + trajec_path
	sim_config["config"]["external-controller"] = True
	if args.allow_pertubations == 1:
		sim_config["config"]["enable-traffic-light"] = True
		sim_config["config"]["test-mode"] = True
		sim_config["config"]["test-file-path"] = os.getcwd() + test_trajec_path
	else:
		sim_config["config"]["enable-traffic-light"] = False
		sim_config["config"]["test-mode"] = False
		sim_config["config"]["test-file-path"] = None

	num_episodes = args.num_episodes
	episode_length = args.episode_length

	exp_name = list(exp_config.keys())[0]
	exp_config[exp_name]["config"]["horizon"] == int(sim_config["config"]["horizon"])

	env = Wrapper(sim_config)
	controller = PPORLController(False, sim_config, exp_config, args.checkpoint_file)

	global_data_dict["num_episodes"] = num_episodes
	global_data_dict["episode-length"] = episode_length
	global_data_dict["time-period"] = sim_config["config"]["time-period"]
	global_data_dict["data"] = {}

	for density in densities:
		
		density_data = {}

		for completed_episodes in range(0, num_episodes):
			
			prev_state = env.reset(density)
			episode_reward = 0.0

			episode_data = {}
			episode_data["cum_reward"] = 0.0
			episode_data["agent_vel"] = []
			episode_data["planner_actions"] = []
			episode_data["comm_actions"] = []

			for step in range(0, episode_length):

				action = controller.getAction(prev_state)
				next_state, reward, done, _ = env.step(action)

				episode_data["planner_actions"].append(env.env.plan_map_reverse[env.env.decoded_action])
				episode_data["comm_actions"].append(env.env.decoded_query)
				agent_idx = [i for i, tup in enumerate(env.env.lane_map_list[env.env.agent_lane]) if tup[env.env.lab2ind["agent"]] == 1][0]
				episode_data["agent_vel"].append(env.env.lane_map_list[env.env.agent_lane][agent_idx][env.env.lab2ind["vel"]])
				
				episode_reward += env.env.before_comm_reward
				prev_state = next_state

				if done:
					break

			episode_data["cum_reward"] = episode_reward
			density_data[completed_episodes] = episode_data

			print("Working for %.1f, Completed Episodes %d/%d"%(density, completed_episodes, num_episodes), end="\r")

		global_data_dict["data"][density] = density_data

	with open(args.save_file_name_path, "wb") as handle:
		pickle.dump(global_data_dict, handle)

	print("Done")
