import os, sys
sys.path.append(os.getcwd() + "/")

import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy

if __name__ == "__main__":

	dataFile = "data_30_1000.pkl"
	path = "/IDM/Local30m/Results"

	with open(os.getcwd()+"/"+path+'/'+dataFile, "rb") as handle:
		data_dict = pickle.load(handle)
	
	
	# Speed Plot
	densities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
	x_axis = np.arange(0, len(densities))

	vel_mean = np.zeros(len(densities))
	vel_std = np.zeros(len(densities))


	for i in range(0, len(densities)):
		
		episodes_avg = np.zeros(data_dict["num-episodes"])

		for episode in range(0, len(data_dict["vel"][i])):
			
			vel_array = np.array(data_dict["vel"][i][episode])
			vel_sum = vel_array.sum() / vel_array.shape[0]
			episodes_avg[episode] = vel_sum * 3.6

		vel_mean[i] = episodes_avg.mean()
		vel_std[i] = episodes_avg.std()

	plt.bar(x_axis, vel_mean, yerr=vel_std, capsize=5)
	plt.xticks(x_axis, densities)
	plt.ylabel("Average Agent Speed (km/hr)")
	plt.xlabel("Traffic Density")
	plt.ylim([0, 9 * 3.6])
	plt.grid()
	plt.savefig("speedLocal.png")
	plt.clf()
	

	
	# Time varying data
	vel_data = np.zeros((len(densities), data_dict["num-episodes"], data_dict["episode-length"]))
	count_vector = np.zeros(len(densities))

	for i in range(0, len(densities)):
		count = 0

		for episode in range(0, len(data_dict["vel"][i])):
			
			if len(data_dict["vel"][i][episode]) < data_dict["episode-length"]:
				pass
			else:
				for step in range(0, data_dict["episode-length"]):
					vel_data[i][episode][step] = data_dict["vel"][i][episode][step]
				count += 1

		count_vector[i] = count

	vel_data = vel_data.sum(axis=1)
	
	for i in range(0, len(densities)):
		vel_data[i] /= count_vector[i]
		vel_data[i] *= 3.6

	#print(vel_data.shape)
	
	for i in range(0, vel_data.shape[0]):
		plt.plot(vel_data[i], label="d=%.1f"%(densities[i]))

	plt.xlabel("time step")
	plt.ylabel("Agent Speed (km/hr)")
	plt.legend()
	plt.ylim([0, 9*3.6])
	plt.show()
	plt.clf()

	#plt.show()
	'''

	# Agent Region Query Pattern
	reg0_count = np.zeros(len(densities))
	reg1_count = np.zeros(len(densities))
	reg2_count = np.zeros(len(densities))
	reg3_count = np.zeros(len(densities))
	null_count = np.zeros(len(densities))


	for i in range(0, len(densities)):

		count = 0

		for episode in range(0, len(data_dict["regions"][i])):
			for step in range(0, len(data_dict["regions"][i][episode])):

				count += 1

				if data_dict["regions"][i][episode][step] == "NULL":
					null_count[i] += 1
				elif data_dict["regions"][i][episode][step] == "reg_0":
					reg0_count[i] += 1
				elif data_dict["regions"][i][episode][step] == "reg_1":
					reg1_count[i] += 1
				elif data_dict["regions"][i][episode][step] == "reg_2":
					reg2_count[i] += 1
				elif data_dict["regions"][i][episode][step] == "reg_3":
					reg3_count[i] += 1
		
		reg0_count[i] /= count
		reg1_count[i] /= count
		reg2_count[i] /= count
		reg3_count[i] /= count
		null_count[i] /= count

	#region_array = np.array((null_count, reg0_count, reg1_count, reg2_count, reg3_count))
	'''