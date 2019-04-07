import os, sys
sys.path.append(os.getcwd() + "/")

from SingleLaneIDM.SimulatorCode.sim import TrafficSim
from SingleLaneIDM.SimulatorCode.observation_queue import ObsQueue
import gym
import yaml
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
import numpy as np
from SingleLaneIDM.SimulatorCode.controllers import ApexRLController, ManualController, PPORLController, PPORLControllerWithActionProbs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--episode-length", type=int, default=1000, help="max lenght of the episode")
parser.add_argument("--num-episodes", type=int, default=100, help="number of epsiodes")
parser.add_argument("--density", type=float, help="vehicle density")
parser.add_argument("--render", default=1, type=int, help="render simulation screen")
parser.add_argument("--config-file", type=str, help="simulation config-file")
parser.add_argument("--idm-only", type=int, default=1, help="Enable only Human Drivers")

class Wrapper(gym.Env):

	def __init__(self, config):
		self.config_file = config
		self.hist_size = config["config"]["hist-size"]
		
		# Initialize the objects here
		self.env = TrafficSim(self.config_file["config"])
		self.action_space = self.env.action_space
		self.observation_space = Box(-float("inf"), float("inf"), shape=(self.hist_size * self.env.observation_space.shape[0],), dtype=np.float)
		self.obs_size = self.env.observation_space.shape[0]
		self.queue = ObsQueue(self.hist_size, self.env.observation_space.shape[0])
		self.queue.resetQueue()

	def reset(self, density=None):
		self.queue.resetQueue()
		obs = self.env.reset(density)
		self.queue.addObs(obs.copy())

		return self.queue.getObs()


	def step(self, action):
		
		obs = self.env.step(action)
		#self.queue.addObs(obs.copy())

		return obs



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
	controller = ManualController(env)
	#controller = RLController(render=False)
	print(env.observation_space)
	print(env.action_space)
	print(env.env.action_map)

	epsiodes = args.num_episodes
	horizon = args.episode_length

	
	for  epsiode in range(0, epsiodes):

		prev_state = env.reset(args.density)

		episode_reward = 0.0

		for step in range(0, horizon):

			#action = controller.getAction(env.queue.queue[-1])
			action = 2
			#action = controller.getAction(prev_state)
			next_state, reward, done, info_dict = env.step(action)

			#print(next_state)
			episode_reward += reward
			prev_state = next_state

			if done:
				break

		print("Episode Lasted for %d time steps and accumulated %.2f Reward"%(step+1, episode_reward))
		print("Time Elapsed : %.2f"%(env.env.time_elapsed))
