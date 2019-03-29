import pickle

file_name = "micro.pkl"

with open(file_name, "rb") as handle:
	data_dict = pickle.load(handle)

copy_data_dict = {}

for density in data_dict.keys():
	copy_data_dict[density] = {}
	
	for lane in data_dict[density].keys():
		copy_data_dict[density][lane] = {}
		copy_data_dict[density][lane]["total_count"] = data_dict[density][lane]["total_count"]
		copy_data_dict[density][lane]["data"] = []

		for traj in range(0,data_dict[density][lane]["total_count"]):

			tmp_list = []

			for idx,vehi in enumerate(data_dict[density][lane]["data"][traj]):
				tuple_list = list(vehi)
				tuple_list.append(idx)
				new_tuple = (tuple_list[0], tuple_list[1], tuple_list[2], tuple_list[3], tuple_list[4])
				tmp_list.append(new_tuple)

		copy_data_dict[density][lane]["data"].append(tmp_list)



with open("micro_with_id.pkl", "wb") as handle:
	pickle.dump(copy_data_dict, handle)

#print(copy_data_dict["0.1"]["lane0"]["data"][0][0])