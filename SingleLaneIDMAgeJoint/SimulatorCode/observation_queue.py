from collections import deque
import numpy as np

class ObsQueue():

	def __init__(self, hist_size, obs_size):
		self.hist_size = hist_size
		self.obs_size = obs_size


	def resetQueue(self):
		self.queue = deque(maxlen=self.hist_size)
		
		tmp_zero_vector = np.zeros(self.obs_size)

		for i in range(0, self.hist_size):
			self.queue.append(tmp_zero_vector.copy())

	def getObs(self):
		return np.array(self.queue).flatten().copy()

	def addObs(self, obs):
		assert obs.shape[0] == self.obs_size
		self.queue.append(obs.copy())
		

if __name__ == "__main__":


	obj = ObsQueue(hist_size=3, obs_size=120)
	obj.resetQueue()
	tmp_vector = np.random.rand(obj.obs_size)
	obj.resetQueue()
	print(obj.getObs())
	obj.addObs(tmp_vector)
	print(obj.getObs())

	obj.addObs(tmp_vector)
	obj.addObs(tmp_vector)
	print(obj.getObs())