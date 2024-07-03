import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy.random as npr
from tqdm import tqdm
from run_PCA import PCA
from utils import coeff2curvature, coeff2position
color = ['C'+str(i) for i in range(10)]


def main():
	folder_name = 'Data/'
	file_name = 'original_data_set'
	data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

	n_elem = data['model']['n_elem']
	L = data['model']['L']
	radius = data['model']['radius']
	s = data['model']['s']
	ds = s[1] - s[0]
	idx_data_pts = data['idx_data_pts']
	input_data = data['input_data']
	true_pos = data['true_pos']
	true_kappa = data['true_kappa']
	output_data_approx = data['output_data_approx']
	pca = data['pca']

	output_size = pca.components.shape[1]
	coeffs = []
	  
	# plt.figure(0)
	for i in tqdm(range(len(input_data))):
		coeff = pca.transform(true_kappa[i])
		coeffs.append(coeff)
		# for j in range(10):
			# plt.scatter([j], coeff[j], color=color[j], s=20, marker='.')
	# plt.show()

	coeffs = np.array(coeffs)
	coeffs_mean = coeffs.mean(axis=0)
	coeffs_std = coeffs.std(axis=0)
	# print(coeffs_mean, coeffs_std)

	npr.seed(2024)
	n_training_data = int(1e5)
	coeffs_rand = npr.randn(n_training_data, output_size) * coeffs_std + coeffs_mean # 100
	# coeffs_rand2 = npr.randn(n_training_data, output_size) * 15 # 100

	# for j in range(10):
	# 	plt.figure(1)
	# 	plt.scatter(np.ones(n_training_data)*j, coeffs_rand[:,j], color=color[j], s=20, marker='.')
	# 	# plt.figure(2)
	# 	# plt.scatter(np.ones(n_training_data)*j, coeffs_rand2[:,j], color=color[j], s=20, marker='.')
	
	# plt.show()

	input_data = []
	true_kappa = []
	true_pos = []

	for i in tqdm(range(n_training_data)):
		kappa = coeff2curvature(coeffs_rand[i], pca)
		pos = coeff2position(coeffs_rand[i], pca)
		plt.figure(1)
		plt.plot(s[1:-1], kappa)
		plt.figure(2)
		plt.plot(pos[0, :], pos[1, :])
		input_data.append(pos[:,idx_data_pts])
		true_pos.append(pos[:,:])
		true_kappa.append(kappa[:])
	
	input_data = np.stack(input_data)
	true_pos = np.stack(true_pos)
	true_kappa = np.stack(true_kappa)

	training_data = {
		'n_training_data': n_training_data,
		'coeffs_rand': coeffs_rand,
		'training_data': input_data,
		'true_pos': true_pos,
        'true_kappa': true_kappa,
		'coeffs': coeffs,
	}
	np.save('Data/training_data_set_1e5.npy', training_data)
	
	plt.show()


if __name__ == "__main__":
	main()