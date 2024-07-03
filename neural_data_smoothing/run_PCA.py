"""
Created on June 10, 2024
@author: Tixian Wang
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from utils import curvature2position
color = ['C'+str(i) for i in range(10)]

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.std = None
    
    def fit(self, X):
        '''
            input X: original data, shape (samples, features)
        '''
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        ## Center the data
        X = (X - self.mean) / self.std
        ## covariance matrix
        cov = np.cov(X.T)
        ## eigenvalues, eigenvectors of the covariance matrix
        eig_val, eig_vec = eig(cov)
        ## Sort eigenvalues 
        idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        print(eig_val[:self.n_components].sum() / eig_val.sum())
        ## feature reductino
        self.components = eig_vec[:,:self.n_components]

    def transform(self, X):
        X = (X - self.mean) / self.std
        X_projected = X @ self.components
        return X_projected

    def reduce(self, X):
        X_hat = self.transform(X) @ self.components.T * self.std + self.mean
        return X_hat
    
    def approximate(self, coeff):
        X_hat = coeff @ self.components.T * self.std + self.mean
        return X_hat

def main():
    folder_name = 'Data/'

    ## simulation data
    n_cases = 12
    # time_skip = 0.1
    step_skip = 2

    ## data point setup
    n_data_pts = 8
    idx_data_pts = np.array([int(100/(n_data_pts-1))*i for i in range(1, n_data_pts-1)]+[-1])

    # print('idx of data pts', idx_data_pts)

    input_data = []
    output_data = []
    true_pos = []
    true_kappa = []

    for i in range(n_cases):
        file_name = 'original_data/original_data'+str(i)
        data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()
        if i == 0:
            n_elem = data['model']['arm']['n_elem']
            L = data['model']['arm']['L']
            radius = data['model']['arm']['radius']
            E = data['model']['arm']['E']
            dt = data['model']['numerics']['step_size']
            s = data['model']['arm']['s']
            s_mean = (s[1:] + s[:-1])/2
            ds = s[1] - s[0]
        t = data['t']
        final_time = t[-1] # data['model']['numerics']['final_time']
        arm = data['arm']
        position = arm[-1]['position'][:,:,:]
        orientation = arm[-1]['orientation']
        kappa = arm[-1]['kappa'][:,:]

        # print(position[::step_skip,:-1,idx_data_pts].shape)
        input_data.append(position[::step_skip,:-1,idx_data_pts])
        output_data.append(kappa[::step_skip, :])
        true_pos.append(position[::step_skip,:-1,:])
        true_kappa.append(kappa[::step_skip, :])

    input_data = np.vstack(input_data) # .reshape(-1, 2*n_data_pts)
    output_data = np.vstack(output_data)
    true_pos = np.vstack(true_pos) # .reshape(-1,2*(n_elem+1))
    true_kappa = np.vstack(true_kappa)
    # print(input_data.shape, output_data.shape, true_pos.shape, true_kappa.shape)

    n_components = 10 # 10 # 5
    pca = PCA(n_components=n_components)
    pca.fit(output_data)
    output_data_approx = pca.reduce(output_data)

    model_data = {
        'n_elem': n_elem,
        'L': L,
        'radius': radius,
        'E': E,
        "dt": dt,
        's': s,
    }

    data = {
        'model': model_data,
        # 'n_data_pts': n_data_pts,
        'idx_data_pts': idx_data_pts,
        'input_data': input_data,
        'output_data_approx': output_data_approx,
        'true_pos': true_pos,
        'true_kappa': true_kappa,
        'pca': pca
    }

    # np.save('Data/original_data_set.npy', data)

    idx_list = [0,1,2,3,4,5]
    # for i in range(len(output_data)):
    for ii in range(len(idx_list)):
        i = idx_list[ii]
        plt.figure(1)
        plt.plot(s[1:-1], output_data[i,:], color=color[ii], ls='-')
        plt.plot(s[1:-1], output_data_approx[i,:], color=color[ii], ls='--')
        plt.figure(2)
        pos_approximate = curvature2position(output_data_approx[i,:])
        plt.plot(pos_approximate[0,:], pos_approximate[1,:], color=color[ii], ls='-')
        plt.plot(true_pos[i, 0,:], true_pos[i, 1,:], color=color[ii], ls='--')
        plt.scatter(input_data[i,0,:], input_data[i,1,:], s=50, marker='o', color=color[ii])

    plt.figure(0)
    for i in range(n_components):
        plt.plot(s[1:-1], pca.components[:,i])

    plt.show()

if __name__ == "__main__":
    main()