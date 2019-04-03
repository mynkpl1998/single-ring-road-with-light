import os, sys
sys.path.append(os.getcwd() + "/")

from SingleLaneIDM.SimulatorCode.main_env import Wrapper 
from SingleLaneIDM.SimulatorCode.main_env import ManualController, ApexRLController, PPORLController
import argparse
import yaml
import numpy as np


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

	env = Wrapper(sim_config)
	controller = PPORLController(False, sim_config, exp_config, args.checkpoint_file)
	print(env.observation_space)
	print(env.action_space)

	episodes = args.num_episodes
	horizon = args.episode_length
	print(horizon)

	for episode in range(0, episodes):

		prev_state = env.reset(args.density)
		episode_reward = 0.0
		#lstm_state = [np.zeros(256), np.zeros(256)]

		for step in range(0, horizon):

			action = controller.getAction(prev_state)
			next_state, reward, done, done_dict = env.step(action)

			episode_reward += reward
			prev_state = next_state

			if done:
				break

		print("Episode Lasted for %d time steps and accumulated %.2f Reward"%(step+1, episode_reward))
