"""
Created on Aug 21, 2024
@author: Tixian Wang
"""
import sys
sys.path.append('../')
import numpy as np
from neural_data_smoothing3D import PCA
from neural_data_smoothing3D import pos_dir_to_input

def main():
	folder_name = 'Data/'
	file_name = 'BR2_arm_data'
	data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

	## data point setup
	n_data_pts = 3 # 5 # exlude the initial point at base
	idx_data_pts = np.array([int(100/(n_data_pts))*i for i in range(1, n_data_pts)]+[-1])
	print('idx of s_j\'s:', idx_data_pts)

	position = data['true_pos']
	director = data['true_dir']

	input_pos = position[..., idx_data_pts]
	input_dir = director[..., idx_data_pts]
	input_data = pos_dir_to_input(input_pos, input_dir)
	# print(position.shape, director.shape, input_pos.shape, input_data.shape)

	flag_save = 0

	if flag_save:
		print('updating data...')
		
		data['n_data_pts'] = n_data_pts
		data['idx_data_pts'] = idx_data_pts
		data['input_data'] = input_data
		np.save('Data/'+file_name+'.npy', data)


if __name__ == "__main__":
	main()