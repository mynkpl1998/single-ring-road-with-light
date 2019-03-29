import os, sys
sys.path.append(os.getcwd() + "/")

from IDM.SimulatorCode.main_env import Wrapper 
from IDM.SimulatorCode.main_env import ManualController, ApexRLController
from tqdm import tqdm
import argparse
import numpy as np
import pickle
import yaml

need to change for lstm

parser = argparse.ArgumentParser()
parser.add_argument("--episode-length", type=int, default=1000, help="max length of the episode")
parser.add_argument("--num-episodes", type=int, default=100, help="number of epsiodes")
parser.add_argument("--checkpoint-file", type=str, help="checkpoint file path")

if __name__ == "__main__":

	args = parser.parse_args()
	
	data_dict = {}

	path = "/SingleLaneIDM/Local30m/ConfigFiles/"
	trajec_path = "/SingleLaneIDM/SimulatorCode/large.pkl"

	sim_config_file_path = os.getcwd() + path + 'ppo-config.yaml'
	exp_config_file_path = os.getcwd() + path + 'ppo-sim-config.yaml'

	with open(sim_config_file_path, "r") as handle:
		sim_config = yaml.load(handle)

	with open(exp_config_file_path, "r") as handle:
		exp_config = yaml.load(handle)

	sim_config["config"]["render"] = False
	sim_config["config"]["trajec-file-path"] = os.getcwd() + trajec_path

	#print(args.render)
	#print(sim_config)
	env = Wrapper(sim_config)
	controller = ApexRLController(False, sim_config, exp_config, args.checkpoint_file)
	print(env.observation_space)
	print(env.action_space)

	episodes = args.num_episodes
	horizon = args.episode_length
	print(horizon)

	densities = env.env.densities
	print(densities)

	velocity_track = []
	action_track = []
	region_query_track = []
	reward_track = []
	
	for i, density in enumerate(densities):

		print("Running for density : ", density)
		velocity_trackd = [] 
		action_trackd = []
		region_query_trackd = []
		reward_trackd = []

		for episode in tqdm(range(0, episodes)):

			velocity_tracke = []
			action_tracke = []
			region_query_tracke = []
			reward_tracke = []

			prev_state = env.reset(density)
			episode_reward = 0.0

			for step in range(0, horizon):

				action = controller.getAction(prev_state)
				next_state, reward, done, done_dict = env.step(action)

				agent_idx = [i for i, tup in enumerate(env.env.lane_map_list[env.env.agent_lane]) if tup[env.env.lab2ind["agent"]] == 1][0]
				
				velocity_tracke.append(env.env.lane_map_list[env.env.agent_lane][agent_idx][env.env.lab2ind["vel"]])
				action_tracke.append(env.env.decoded_action)
				#region_query_tracke.append(env.env.decoded_query)
				reward_tracke.append(reward)

				episode_reward += reward
				prev_state = next_state

				if done:
					break
			
			velocity_trackd.append(velocity_tracke)
			action_trackd.append(action_tracke)
			#region_query_trackd.append(region_query_tracke)
			reward_trackd.append(reward_tracke)


		velocity_track.append(velocity_trackd)
		action_track.append(action_trackd)
		#region_query_track.append(region_query_trackd)
		reward_track.append(reward_trackd)
	
			#print("Episode Lasted for %d time steps and accumulated %.2f Reward"%(step+1, episode_reward))

	
	data_dict["vel"] = velocity_track
	data_dict["actions"] = action_track
	#data_dict["regions"] = region_query_track
	data_dict["reward"] = reward_track
	data_dict["num-episodes"] = args.num_episodes
	data_dict["episode-length"] = args.episode_length

	print(data_dict.keys())

	filename = "data_"+str(args.num_episodes)+"_"+str(args.episode_length)+".pkl"
	
	with open(filename, "wb") as handle:
		pickle.dump(data_dict, handle)

