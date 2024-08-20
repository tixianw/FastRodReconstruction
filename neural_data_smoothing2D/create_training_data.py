"""
Created on Jun 18, 2024
@author: Tixian Wang
"""
import numpy as np
import matplotlib.pyplot as plt
# import torch
import numpy.random as npr
from tqdm import tqdm
from run_PCA import PCA
from utils import coeff2curvature, coeff2position
color = ['C'+str(i) for i in range(20)]


def main():
	folder_name = 'Data/'
	file_name = 'fake_br2_arm_data1' # 'pyelastica_arm_data2' # 
	data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

	n_elem = data['model']['n_elem']
	L = data['model']['L']
	radius = data['model']['radius']
	s = data['model']['s']
	ds = s[1] - s[0]
	idx_data_pts = data['idx_data_pts']
	input_data = data['input_data']
	output_data = data['output_data']
	true_pos = data['true_pos']
	true_kappa = data['true_kappa']
	pca = data['pca']

	output_size = pca.components.shape[1] # + 1
	coeffs = pca.transform(output_data) # np.hstack([output_data[..., [0]], pca.transform(output_data[..., 1:])])
	plt.figure(0)
	for j in range(min(10, coeffs.shape[1])):
		plt.scatter([j]*len(coeffs), coeffs[:,j], color=color[j], s=20, marker='.')
	# plt.show()
	# quit()

	coeffs_mean = coeffs.mean(axis=0)
	coeffs_std = coeffs.std(axis=0)
	coeffs_low = coeffs.min(axis=0)
	coeffs_high = coeffs.max(axis=0)
	npr.seed(2024)
	n_training_data = int(1e5)
	coeffs_rand = npr.randn(n_training_data, output_size) * coeffs_std + coeffs_mean
	# coeffs_rand = npr.uniform(coeffs_low, coeffs_high, size=(n_training_data, output_size))

	for j in range(min(10, coeffs.shape[1])):
		plt.figure(1)
		plt.scatter([j]*n_training_data, coeffs_rand[:,j], color=color[j], s=20, marker='.')
		# plt.figure(2)
		# plt.scatter(np.ones(n_training_data)*j, coeffs_rand2[:,j], color=color[j], s=20, marker='.')
	
	plt.show()

	kappa_rand = coeff2curvature(coeffs_rand, pca)
	pos_rand = coeff2position(coeffs_rand, pca)

	# print(pos_rand[..., idx_data_pts].shape, pos_rand.shape, kappa_rand.shape)

	# np.random.seed(2024)
	idx_list = np.random.randint(len(true_kappa), size=10) # [i*10 for i in range(10)]
	for ii in range(len(idx_list)):
		i = idx_list[ii]
		plt.figure(2)
		plt.plot(s[1:-1], kappa_rand[i])
		plt.figure(3)
		plt.plot(pos_rand[i, 0, :], pos_rand[i, 1, :])

	training_data = {
		'n_training_data': n_training_data,
		'coeffs_rand': coeffs_rand,
		'training_data': pos_rand[..., idx_data_pts],
		'true_pos': pos_rand,
        'true_kappa': kappa_rand,
		'coeffs': coeffs,
	}
	np.save('Data/training_data_set1.npy', training_data) # training_data_set1
	
	plt.show()


if __name__ == "__main__":
	main()