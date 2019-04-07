import os, sys
sys.path.append(os.getcwd() + "/")

from SingleLaneIDM.SimulatorCode.main_env import Wrapper 
from SingleLaneIDM.SimulatorCode.main_env import ManualController, ApexRLController, PPORLController
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle
import yaml
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--episode-length", type=int, default=3000, help="max length of the episode")
parser.add_argument("--num-episodes", type=int, default=10, help="number of epsiodes")
parser.add_argument("--checkpoint-file", type=str, help="checkpoint file path")
parser.add_argument("--sim-config-file", type=str, help="simulation configuration file")
parser.add_argument("--ppo-config-file", type=str, help="ppo experiment configuration file")
parser.add_argument("--save-path", type=str, help="file name for saving simulated data")
parser.add_argument("--case-name", type=str, help="case name used to create dataset and folder")

if __name__ == "__main__":

	args = parser.parse_args()

	dirname = args.save_path + args.case_name
	
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	if not os.path.exists(dirname + "/" + "Images"):
		os.makedirs(dirname + "/" + "Images")

	trajec_path = "/SingleLaneIDM/SimulatorCode/micro.pkl"

	global_data_dict = {}
	densities = [0.2, 0.4, 0.5, 0.7]
	
	with open(args.sim_config_file, "r") as handle:
		sim_config = yaml.load(handle)

	with open(args.ppo_config_file, "r") as handle:
		exp_config = yaml.load(handle)

	sim_config["config"]["render"] = True

	sim_config["config"]["trajec-file-path"] = os.getcwd() + trajec_path
	sim_config["config"]["external-controller"] = True
	sim_config["config"]["enable-traffic-light"] = True
	sim_config["config"]["enable-frame-capture"] = True

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

		completed_episodes = 0
		while completed_episodes < num_episodes:
			
			prev_state = env.reset(density)
			episode_reward = 0.0
			successful_episode = False

			episode_data = {}
			episode_data["cum_reward"] = 0.0
			episode_data["agent_vel"] = np.zeros(episode_length)
			episode_data["planner_actions"] = []
			episode_data["comm_actions"] = []

			file_name = args.save_path + args.case_name + "/Images/" + str(env.env.num_cars_in_setup) + "_%d.png"%(0)
			plt.imsave(file_name, env.env.curr_screen, )

			for step in range(0, episode_length):

				action = controller.getAction(prev_state)
				next_state, reward, done, _ = env.step(action)

				episode_data["planner_actions"].append(env.env.plan_map_reverse[env.env.decoded_action])
				episode_data["comm_actions"].append(env.env.decoded_query)
				agent_idx = [i for i, tup in enumerate(env.env.lane_map_list[env.env.agent_lane]) if tup[env.env.lab2ind["agent"]] == 1][0]
				episode_data["agent_vel"][step] = env.env.lane_map_list[env.env.agent_lane][agent_idx][env.env.lab2ind["vel"]]
				file_name = args.save_path + args.case_name + "/Images/" + str(env.env.num_cars_in_setup) + "_%d.png"%(env.env.num_steps)
				plt.imsave(file_name, env.env.curr_screen, )
				
				episode_reward += reward
				prev_state = next_state

				if done:
					break

			if (step+1) == episode_length:
				completed_episodes += 1
				successful_episode = True

			if successful_episode:
				episode_data["cum_reward"] = episode_reward
				density_data[completed_episodes-1] = episode_data

			print("Working for %.1f, Completed Episodes %d/%d"%(density, completed_episodes, num_episodes), end="\r")

		global_data_dict["data"][density] = density_data

	with open(dirname + "/dataset.pkl", "wb") as handle:
		pickle.dump(global_data_dict, handle)

	print("Done")