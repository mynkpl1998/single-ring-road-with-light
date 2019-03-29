import os, sys
sys.path.append(os.getcwd() + "/")

from SingleLaneIDM.SimulatorCode.main_env import Wrapper 
from SingleLaneIDM.SimulatorCode.main_env import ManualController, ApexRLController, PPORLController
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle
import yaml
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--episode-length", type=int, default=3000, help="max length of the episode")
parser.add_argument("--checkpoint-file", type=str, help="checkpoint file path")
parser.add_argument("--density", type=float, help="vehicular density")
parser.add_argument("--render", type=int, help="renders environment to screen")

def create_file_name(config_file):
	file_name = 'view_size_'
	file_name += str(config_file["config"]["view-size"]) + 'm_'
	file_name += 'comm_'
	file_name += str(config_file["config"]["comm-mode"]) + '_'
	file_name += "extr_ctrl_"
	file_name += str(config_file["config"]["external-controller"])
	return file_name


if __name__ == "__main__":

	args = parser.parse_args()

	path = "/SingleLaneIDM/Local30m/ConfigFiles/"
	trajec_path = "/SingleLaneIDM/SimulatorCode/micro.pkl"

	sim_config_file_path = os.getcwd() + path + 'ppo-sim-config.yaml'
	exp_config_file_path = os.getcwd() + path + 'ppo-config.yaml'

	with open(sim_config_file_path, "r") as handle:
		sim_config = yaml.load(handle)

	with open(exp_config_file_path, "r") as handle:
		exp_config = yaml.load(handle)

	if args.render == 1:
		sim_config["config"]["render"] = True
	else:
		sim_config["config"]["render"] = False

	sim_config["config"]["external-controller"] = True
	sim_config["config"]["trajec-file-path"] = os.getcwd() + trajec_path
	file_name = create_file_name(sim_config)


	env = Wrapper(sim_config)
	controller = PPORLController(False, sim_config, exp_config, args.checkpoint_file)
	print(env.observation_space)
	print(env.action_space)
	horizon = args.episode_length
	episodes = 1

	
	for episode in range(0, episodes):

		prev_state = env.reset(args.density)
		track_list = env.env.track_map_list
		vel_list = env.env.track_vel_list
		pos_dict = {}
		vel_dict = {}

		for veh in track_list[0].keys():
			pos_dict[veh] = []

		for veh in vel_list[0].keys():
			vel_dict[veh] = []
		
		episode_reward = 0.0
		lstm_state = [np.zeros(256), np.zeros(256)]


		for step in range(0, horizon):

			action, lstm_state = controller.getAction(prev_state, lstm_state)
			next_state, reward, done, done_dict = env.step(action)

			track_list = env.env.track_map_list
			vel_list = env.env.track_vel_list
			
			for veh in track_list[0].keys():
				pos_dict[veh].append(track_list[0][veh])

			for veh in vel_list[0].keys():
				vel_dict[veh].append(vel_list[0][veh])
			
			episode_reward += reward
			prev_state = next_state
			
			if done:
				break

		print("Episode Lasted for %d time steps and accumulated %.2f Reward"%(step+1, episode_reward))

	
	for veh in pos_dict:
		#print("veh id : ", veh,", final pos : ", pos_dict[veh][-1])
		plt.plot(pos_dict[veh])
		plt.title("Num Vehicles : %d"%(len(pos_dict)))
		plt.grid()

	plt.savefig(file_name+".png")

	plt.clf()
	
	
	for veh in vel_dict:
		plt.plot(vel_dict[veh])
		plt.title("Num Vehicles : %d"%(len(vel_dict)))
		plt.grid()
		
	#plt.plot(vel_dict[env.env.agent_id])

	plt.savefig(file_name+"_vel"+".png")