from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, sys
sys.path.append(os.getcwd() + "/")
import tensorflow as tf
from scipy.special import softmax

class ManualController():

	def __init__(self, env_obj):
		self.obs_size = env_obj.obs_size
		self.env_obj = env_obj

	def getAction(self, state):
		assert state.shape[0] == self.obs_size

		agent_lane = self.env_obj.env.agent_lane
		occ_shape = self.env_obj.env.occ_grid.shape
		
		'''
		shaped_occ = state.reshape(occ_shape[0], occ_shape[1])
		occ_grid = shaped_occ[0:3, :]
		#vel_grid = shaped_occ[3:, :]
		agent_lane_occ = occ_grid[agent_lane]

		'''
		return 0


class ApexRLController():

	def __init__(self, render, sim_config, exp_config, checkpoint_path):
		
		import ray
		from ray.tune import run_experiments
		from ray.tune.registry import register_env
		from ray import tune
		import yaml
		from SingleLaneIDM.SimulatorCode.main_env import Wrapper

		from ray.rllib.agents.dqn.apex import ApexAgent
		import ray.rllib.agents.dqn as apex

		import os
		import pickle

		if render == 1:
			sim_config["config"]["render"] = True
		else:
			sim_config["config"]["render"] = False

		sim_config["config"]["acc-noise"] = False

		exp_name = list(exp_config.keys())[0]

		exp_config[exp_name]["config"]["num_gpus"] = 0
		exp_config[exp_name]["config"]["num_workers"] = 2
		exp_config[exp_name]["config"]["num_envs_per_worker"] = 1

		env_creator_name = "tsim-v0"
		register_env(env_creator_name, lambda config: Wrapper(sim_config))

		ray.init()
		self.agent = ApexAgent(env="tsim-v0", config=exp_config[exp_name]["config"])
		self.agent.restore(checkpoint_path)
		self.agent.optimizer.foreach_evaluator(lambda ev: ev.for_policy(lambda pi:pi.set_epsilon(0.0), policy_id="default"))

	
	def getAction(self, state):
		return self.agent.compute_action(state)

class PPORLController():

	def __init__(self, render, sim_config, exp_config, checkpoint_path):
		
		import ray
		from ray.tune import run_experiments
		from ray.tune.registry import register_env
		from ray import tune
		import yaml
		from SingleLaneIDM.SimulatorCode.main_env import Wrapper

		from ray.rllib.agents.ppo.ppo import PPOAgent
		import os
		import pickle

		if render == 1:
			sim_config["config"]["render"] = True
		else:
			sim_config["config"]["render"] = False

		sim_config["config"]["acc-noise"] = False

		exp_name = list(exp_config.keys())[0]
		exp_config[exp_name]["config"]["num_gpus"] = 0
		exp_config[exp_name]["config"]["num_workers"] = 1
		exp_config[exp_name]["config"]["num_envs_per_worker"] = 1

		env_creator_name = "tsim-v0"
		register_env(env_creator_name, lambda config: Wrapper(sim_config))

		ray.init()
		self.agent = PPOAgent(env="tsim-v0", config=exp_config[exp_name]["config"])
		self.agent.restore(checkpoint_path)
		#self.agent.optimizer.foreach_evaluator(lambda ev: ev.for_policy(lambda pi:pi.set_epsilon(0.0), policy_id="default"))

	
	def getAction(self, state):

		action = self.agent.compute_action(state)
		#print(action)
		return action

class PPORLControllerWithActionProbs():

	def __init__(self, render, sim_config, exp_config, checkpoint_path):
		
		import ray
		from ray.tune import run_experiments
		from ray.tune.registry import register_env
		from ray import tune
		import yaml
		from SingleLaneIDM.SimulatorCode.main_env import Wrapper

		from ray.rllib.agents.ppo.ppo import PPOAgent
		import os
		import pickle

		if render == 1:
			sim_config["config"]["render"] = True
		else:
			sim_config["config"]["render"] = False

		sim_config["config"]["acc-noise"] = False

		exp_name = list(exp_config.keys())[0]
		exp_config[exp_name]["config"]["num_gpus"] = 0
		exp_config[exp_name]["config"]["num_workers"] = 1
		exp_config[exp_name]["config"]["num_envs_per_worker"] = 1

		env_creator_name = "tsim-v0"
		register_env(env_creator_name, lambda config: Wrapper(sim_config))

		ray.init()
		self.agent = PPOAgent(env="tsim-v0", config=exp_config[exp_name]["config"])
		self.agent.restore(checkpoint_path)
		#self.agent.optimizer.foreach_evaluator(lambda ev: ev.for_policy(lambda pi:pi.set_epsilon(0.0), policy_id="default"))

	
	def getAction(self, observation):
		
		state = []
		preprocessed = self.agent.local_evaluator.preprocessors["default"].transform(observation)
		filtered_obs = self.agent.local_evaluator.filters['default'](preprocessed, update=False)
		
		result = self.agent.get_policy('default').compute_single_action(filtered_obs, state, None, None, None, clip_actions=self.agent.config["clip_actions"])
		
		action = result[0]
		probs = softmax(result[2]["logits"]) # imported from scipy.special
		
		return action, probs
