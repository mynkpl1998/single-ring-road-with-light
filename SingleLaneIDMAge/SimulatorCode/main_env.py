import os, sys
sys.path.append(os.getcwd() + "/")

from SingleLaneIDMAge.SimulatorCode.sim import TrafficSim
from SingleLaneIDMAge.SimulatorCode.observation_queue import ObsQueue
import gym
import yaml
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
import numpy as np
from SingleLaneIDMAge.SimulatorCode.controllers import ApexRLController, ManualController, PPORLController
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--episode-length", type=int, default=1000, help="max lenght of the episode")
parser.add_argument("--num-episodes", type=int, default=100, help="number of epsiodes")
parser.add_argument("--density", type=float, help="vehicle density")
parser.add_argument("--render", default=1, type=int, help="render simulation screen")
parser.add_argument("--config-file", type=str, help="simulation config-file")
parser.add_argument("--idm-only", type=int, default=1, help="Enable only Human Drivers")

class Wrapper(MultiAgentEnv):

	def __init__(self, config):
		self.config_file = config
		self.hist_size = config["config"]["hist-size"]
		
		# Initialize the objects here
		self.env = TrafficSim(self.config_file["config"])
		self.planner_action_space = self.env.planner_action_space
		self.comm_action_space = self.env.comm_action_space
		
		# Observation Space
		self.planner_observation_space = Box(-float("inf"), float("inf"), shape=(self.hist_size * self.env.planner_observation_space.shape[0],), dtype=np.float)
		self.comm_observation_space = Box(-float("inf"), float("inf"), shape=(self.hist_size * self.env.comm_observation_space.shape[0],), dtype=np.float)

		self.planner_obs_size = self.env.planner_observation_space.shape[0]
		self.comm_obs_size = self.env.comm_observation_space.shape[0]

		self.planner_queue = ObsQueue(self.hist_size, self.env.planner_observation_space.shape[0])
		self.comm_queue = ObsQueue(self.hist_size, self.env.comm_observation_space.shape[0])

		self.planner_queue.resetQueue()
		self.comm_queue.resetQueue()

	def reset(self, density=None):

		self.planner_queue.resetQueue()
		self.comm_queue.resetQueue()

		curr_obs = self.env.reset(density)
		self.planner_queue.addObs(curr_obs["planner"].copy())
		self.comm_queue.addObs(curr_obs["comm"].copy())

		obs = {}
		obs["planner"] = self.planner_queue.getObs()
		obs["comm"] = self.comm_queue.getObs()

		return obs


	def step(self, action):

		obs, reward, dones, info = self.env.step(action)
		
		self.planner_queue.addObs(obs["planner"].copy())
		self.comm_queue.addObs(obs["comm"].copy())

		new_obs = {}
		new_obs["planner"] = self.planner_queue.getObs()
		new_obs["comm"] = self.comm_queue.getObs()

		return new_obs, reward, dones, info


if __name__ == "__main__":

	args = parser.parse_args()
	
	trajec_path = "/SingleLaneIDM/SimulatorCode/micro.pkl"

	sim_config_file_path = args.config_file

	with open(sim_config_file_path, "r") as handle:
		sim_config = yaml.load(handle)

	if args.render == 1:
		sim_config["config"]["render"] = True
	else:
		sim_config["config"]["render"] = False

	sim_config["config"]["trajec-file-path"] = os.getcwd() + trajec_path

	if args.idm_only == 1:
		sim_config["config"]["external-controller"] = False
	else:
		sim_config["config"]["external-controller"] = True


	env = Wrapper(sim_config)
	#controller = RLController(render=False)
	
	print("Observation Spaces : ")
	print("PLAN : ", env.planner_observation_space)
	print("COMM : ", env.comm_observation_space)

	print("Action Spaces : ")
	print("PLAN : ", env.planner_action_space)
	print("COMM : ", env.comm_action_space)

	epsiodes = args.num_episodes
	horizon = args.episode_length

	
	for  epsiode in range(0, epsiodes):

		prev_state = env.reset(args.density)
		#print(prev_state)

		episode_reward = 0.0

		for step in range(0, horizon):

			act = {}
			act["comm"] = np.random.randint(0, env.comm_action_space.n)
			act["planner"] = np.random.randint(0, env.planner_action_space.n)

			obs, rew, done, info = env.step(act)
			vehicles = env.env.lane_map_list[0]

			vec = []
			for vehicle in vehicles:
				vec.append(vehicle[env.env.lab2ind["vel"]])
			print("Step : ", step+1, vec)
			
			if done["__all__"]:
				break

		print("Episode Lasted for %d time steps and accumulated %.2f Reward"%(step+1, episode_reward))