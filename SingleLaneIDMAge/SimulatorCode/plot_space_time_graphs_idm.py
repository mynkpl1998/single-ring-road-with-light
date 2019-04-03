import os, sys
sys.path.append(os.getcwd() + "/")

from SingleLaneIDM.SimulatorCode.sim import TrafficSim
import matplotlib.pyplot as plt
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--episode-length", type=int, default=3000, help="max length of the episode")
parser.add_argument("--density", type=float, help="vehiclar density")
parser.add_argument("--config-file", type=str, help="simulation config file")
parser.add_argument("--render", type=int, default=1, help="renders simulation screen")

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

	with open(args.config_file, "r") as handle:
		sim_config = yaml.load(handle)

	sim_config["config"]["external-controller"] = False
	trajec_path = "/IDM/SimulatorCode/micro.pkl"
	sim_config["config"]["trajec-file-path"] = os.getcwd() + trajec_path

	if args.render == 1:
		sim_config["config"]["render"] = True
	else:
		sim_config["config"]["render"] = False

	file_name = create_file_name(sim_config)
	
	env = TrafficSim(sim_config["config"])

	for i in range(0, 1):
		state = env.reset(args.density)
		done = False
		total_reward = 0.0
		track_list = env.track_map_list
		vel_list = env.track_vel_list
		pos_dict = {}
		vel_dict = {}

		for veh in track_list[0].keys():
			pos_dict[veh] = []

		for veh in vel_list[0].keys():
			vel_dict[veh] = []


		for step in range(0, args.episode_length):        
			state, reward, done, _ = env.step(0)

			track_list = env.track_map_list
			vel_list = env.track_vel_list

			for veh in track_list[0].keys():
				pos_dict[veh].append(track_list[0][veh])

			for veh in vel_list[0].keys():
				vel_dict[veh].append(vel_list[0][veh])

	for veh in pos_dict:
		plt.plot(pos_dict[veh])
		plt.title("Num Vehicles : %d"%(len(pos_dict)))
		plt.grid()

	plt.savefig(file_name+'.png')

	plt.clf()
	
	for veh in vel_dict:
		plt.plot(vel_dict[veh])
		plt.title("Num Vehicles : %d"%(len(vel_dict)))
		plt.grid()
	#plt.plot(vel_dict[env.agent_id])

	plt.savefig(file_name+"_vel"+".png")

	

    

