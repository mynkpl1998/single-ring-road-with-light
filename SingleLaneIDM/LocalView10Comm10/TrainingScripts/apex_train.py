from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray import tune
from ray.tune.logger import pretty_print

import yaml
import os, sys
sys.path.append(os.getcwd() + "/")

from SingleLaneIDM.SimulatorCode.main_env import Wrapper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sim-config-file", type=str, help="simulation configuration file")
parser.add_argument("--apex-config-file", type=str, help="ppo experiment configuration file")

if __name__ == "__main__":

	args = parser.parse_args()
	trajec_path = "/SingleLaneIDM/SimulatorCode/micro.pkl"
	
	with open(args.sim_config_file, "r") as handle:
		sim_config = yaml.load(handle)

	with open(args.apex_config_file, "r") as handle:
		exp_config = yaml.load(handle)


	sim_config["config"]["render"] = False
	sim_config["config"]["render-grid"] = False
	sim_config["config"]["trajec-file-path"] = os.getcwd() + trajec_path
	sim_config["config"]["external-controller"] = True

	exp_name = list(exp_config.keys())[0]

	exp_config[exp_name]["config"]["callbacks"]["on_episode_end"] = None
	exp_config[exp_name]["config"]["horizon"] = int(sim_config["config"]["horizon"])

	env_creator_name = "tsim-v0"
	register_env(env_creator_name, lambda config: Wrapper(sim_config))
		
	#print(exp_config)
	ray.init()
	run_experiments(exp_config)