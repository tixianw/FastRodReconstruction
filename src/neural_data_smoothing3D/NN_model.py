"""
Created on Aug 21, 2024
@author: Tixian Wang
"""

import sys

sys.path.append("../")
import numpy as np
import numpy.random as npr
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# from neural_data_smoothing3D import PCA
from neural_data_smoothing3D.utils import (
    _aver,
    coeff2posdir_torch,
    coeff2strain_torch,
)


class TensorConstants:
    def __init__(
        self,
        bend_twist_stiff,
        idx_data_pts,
        dl,
        chi_r,
        chi_d,
        pca,
        input_size,
        output_size,
    ):
        self.bend_twist_stiff = torch.from_numpy(bend_twist_stiff).float()
        self.idx_data_pts = torch.from_numpy(idx_data_pts)
        self.dl = torch.from_numpy(dl).float()
        nominal_shear = np.vstack([np.zeros([2, 100]), np.ones(100)])
        self.nominal_shear = torch.from_numpy(nominal_shear).float()
        self.chi_r = torch.from_numpy(chi_r).float() # chi_r
        self.chi_d = torch.from_numpy(chi_d).float() # chi_d
        pca_mean = np.vstack([pca[i].mean for i in range(len(pca))])
        pca_std = np.vstack([pca[i].std for i in range(len(pca))])
        n_components = np.array([pca[i].n_components for i in range(len(pca))])
        self.n_components = torch.from_numpy(n_components)
        self.pca_mean = torch.from_numpy(pca_mean).float()
        self.pca_std = torch.from_numpy(pca_std).float()
        pca_components = np.hstack([pca[i].components for i in range(len(pca))])
        self.pca_components = torch.from_numpy(pca_components).float()
        self.input_size = input_size
        self.output_size = output_size


class CustomLoss(nn.Module):
    def __init__(self, tensor_constants):
        super().__init__()
        self.tensor_constants = tensor_constants

    def forward(self, outputs, labels):
        kappa_hat = coeff2strain_torch(outputs, self.tensor_constants)
        diff = kappa_hat - labels
        loss = 0.5 * torch.sum(
            diff * diff * _aver(self.tensor_constants.dl), axis=(1, 2)
        )  # *self.tensor_constants.EI
        return loss.mean()


class CurvatureSmoothing3DLoss(nn.Module):
    def __init__(self, tensor_constants):
        super().__init__()
        self.tensor_constants = tensor_constants

    def potential_energy(self):
        V = 0.5 * torch.sum(
            torch.einsum(
                "nik,ijk,njk->nk",
                self.kappa_hat,
                self.tensor_constants.bend_twist_stiff,
                self.kappa_hat,
            )
            * _aver(self.tensor_constants.dl),
            axis=1,
        )
        return V.mean()

    def data_matching_cost(self, inputs, outputs):
        input_pos = inputs[:, :3, :]
        input_dir = torch.stack(
            [
                inputs[:, 3:6, :],
                torch.cross(inputs[:, 6:9, :], inputs[:, 3:6, :], axis=1),
                inputs[:, 6:9, :],
            ],
            axis=2,
        )
        # inputs = torch.flatten(inputs, start_dim=1)
        pos_dir = coeff2posdir_torch(outputs, self.tensor_constants)
        pos_difference = input_pos - pos_dir[0] #  torch.flatten(input_pos - pos_dir[0], start_dim=1)
        dir_difference = input_dir - pos_dir[1] # torch.flatten(input_dir - pos_dir[1], start_dim=1)
        Phi = 0.5 * torch.sum(
            pos_difference * pos_difference * self.tensor_constants.chi_r, axis=(1,2)
        ) + 0.5 * torch.sum(
            dir_difference * dir_difference * self.tensor_constants.chi_d, axis=(1,2,3)
        )
        return Phi.mean()

    def forward(self, outputs, inputs):
        """
        input_data: discrete position data points
        outputs: weights of PCA compunents for approximated targ et curvature
        """
        # inputs = torch.flatten(inputs)
        # outputs = torch.flatten(outputs)
        # self.kappa_hat = torch.mv(self.tensor_constants.pca_components, outputs)
        self.kappa_hat = coeff2strain_torch(outputs, self.tensor_constants)
        J = self.potential_energy()
        # print('V:', J.data, 'Phi:', self.data_matching_cost(inputs, outputs).data)
        J += self.data_matching_cost(inputs, outputs)
        # J = self.data_matching_cost(inputs, outputs)
        return J


class CurvatureSmoothing3DNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = (
            nn.SiLU()
        )  # nn.ReLU() # nn.GELU() # convergence speed: GELU > SiLU > ReLU
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            self.activation,
            nn.Linear(32, 16),
            self.activation,
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.linear_relu_stack(x)
        return x


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CurvatureSmoothing3DModel:
    def __init__(
        self,
        tensor_constants,
        input_data,
        num_epochs=1,
        batch_size=128,
        eval_batch_size=100,
        labels=None,
    ):
        self.tensor_constants = tensor_constants
        # self.input_size = len(tensor_constants.idx_data_pts) * 2
        # self.output_size = tensor_constants.pca_components.shape[1] # + 1
        self.input_size = tensor_constants.input_size
        self.output_size = tensor_constants.output_size
        self.net = CurvatureSmoothing3DNet(self.input_size, self.output_size)
        self.loss_fn = CurvatureSmoothing3DLoss(
            tensor_constants
        )  # CustomLoss(tensor_constants) #
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.input_data = torch.from_numpy(input_data).float()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.labels = labels
        self.train_valid_test_split()
        self.train_losses = []
        self.validation_losses = []

    def train_valid_test_split(self):
        # training_data = torch.from_numpy(self.input_data).float()
        # training_dataset = MyDataset(training_data, self.labels)
        generator = torch.Generator().manual_seed(2024)
        training_set, validation_set, test_set = random_split(
            self.input_data, [0.8, 0.1, 0.1], generator=generator
        )
        self.training_loader = DataLoader(
            training_set, batch_size=self.batch_size, shuffle=True
        )
        self.validation_loader = DataLoader(
            validation_set, batch_size=self.eval_batch_size, shuffle=True
        )
        self.test_loader = DataLoader(test_set, batch_size=len(test_set))
        self.num_train_interation = len(self.training_loader)
        print(
            "# training samples:",
            len(training_set),
            "# validation samples:",
            len(validation_set),
            "# test samples:",
            len(test_set),
        )

    ### Train the model
    def train_one_epoch(self, epoch_idx, steps_to_check=100):
        running_loss = 0.0
        last_loss = 0.0
        for i, inputs in enumerate(self.training_loader):  # (inputs, labels)
            outputs = self.net(inputs)
            loss = self.loss_fn(outputs, inputs)  # loss_fn(outputs, labels) #
            self.optimizer.zero_grad()
            loss.backward()  # (retain_graph=True)
            self.optimizer.step()
            running_loss += loss.item()
            if (
                i % steps_to_check == steps_to_check - 1
                or i == self.num_train_interation - 1
            ):
                last_loss = running_loss / (i % steps_to_check + 1)
                print(
                    f"Epoch {epoch_idx+1}/{self.num_epochs}, Step [{i+1}/{self.num_train_interation}], Loss: {last_loss:.8f}"
                )
                running_loss = 0.0
                self.train_losses.append(last_loss)
    
    def model_train(self, file_name, check_epoch_idx=10):
        for epoch_idx in range(self.num_epochs):
            ## Make sure gradient tracking is on, and do a pass over the data
            self.net.train(True)
            self.train_one_epoch(epoch_idx)

            running_vloss = 0.0
            self.net.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vinputs in enumerate(self.validation_loader):
                    voutputs = self.net(vinputs)
                    vloss = self.loss_fn(voutputs, vinputs)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            self.validation_losses.append(avg_vloss)
            print(
                f"Losses: train {self.train_losses[-1]:.8f} valid {avg_vloss:.8f}"
            )

            if (epoch_idx+1)%check_epoch_idx==0 or ((epoch_idx+1)%check_epoch_idx and epoch_idx==self.num_epochs-1):

                self.test_loss = 0.0
                for i, test_inputs in enumerate(self.test_loader):
                    test_outputs = self.net(test_inputs)
                    t_loss = self.loss_fn(test_outputs, test_inputs)
                    self.test_loss += t_loss.item()
                self.test_loss /= i + 1
                print(f"test loss at epoch {epoch_idx:d}: {self.test_loss:.8f}")
                self.model_save(file_name, epoch_idx)

    def model_save(self, file_name, epoch_idx):
        torch.save(
            {
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                'current_epoch': epoch_idx,
                "tensor_constants": self.tensor_constants,
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "losses": [
                    self.train_losses,
                    self.validation_losses,
                    self.test_loss,
                ],
            },
            file_name+'_epoch%03d'%(epoch_idx+1)+'.pt',
        )


    # def model_train(self):
    #     for epoch_idx in range(self.num_epochs):
    #         ## Make sure gradient tracking is on, and do a pass over the data
    #         self.net.train(True)
    #         self.train_one_epoch(epoch_idx)

    #         running_vloss = 0.0
    #         self.net.eval()
    #         # Disable gradient computation and reduce memory consumption.
    #         with torch.no_grad():
    #             for i, vinputs in enumerate(self.validation_loader):
    #                 voutputs = self.net(vinputs)
    #                 vloss = self.loss_fn(voutputs, vinputs)
    #                 running_vloss += vloss.item()

    #         avg_vloss = running_vloss / (i + 1)
    #         self.validation_losses.append(avg_vloss)
    #         print(
    #             f"Losses: train {self.train_losses[-1]:.8f} valid {avg_vloss:.8f}"
    #         )

    #     self.test_loss = 0.0
    #     for i, test_inputs in enumerate(self.test_loader):
    #         test_outputs = self.net(test_inputs)
    #         t_loss = self.loss_fn(test_outputs, test_inputs)
    #         self.test_loss += t_loss.item()
    #     self.test_loss /= i + 1
    #     print(f"test loss: {self.test_loss:.8f}")

    # def model_save(self, file_name):
    #     torch.save(
    #         {
    #             "batch_size": self.batch_size,
    #             "num_epochs": self.num_epochs,
    #             "tensor_constants": self.tensor_constants,
    #             "model": self.net.state_dict(),
    #             "optimizer": self.optimizer.state_dict(),
    #             "losses": [
    #                 self.train_losses,
    #                 self.validation_losses,
    #                 self.test_loss,
    #             ],
    #         },
    #         file_name,
    #     )


def main():
    # print('Cuda is available?', torch.cuda.is_available())
    # # torch.cuda.device_count()
    # # torch.cuda.current_device()
    # torch.autograd.set_detect_anomaly(True)

    folder_name = "Data/"
    file_name = "BR2_arm_data"  # 'pyelastica_arm_data' #
    data = np.load(folder_name + file_name + ".npy", allow_pickle="TRUE").item()

    n_elem = data["model"]["n_elem"]
    L = data["model"]["L"]
    radius = data["model"]["radius"]
    s = data["model"]["s"]
    dl = data["model"]["dl"]
    nominal_shear = data["model"]["nominal_shear"]
    idx_data_pts = data["idx_data_pts"]
    input_data = data["input_data"]
    true_pos = data["true_pos"]
    true_dir = data["true_dir"]
    true_kappa = data["true_kappa"]
    pca = data["pca"]

    training_data = np.load(
        folder_name + "training_data_set.npy", allow_pickle="TRUE"
    ).item()  # training_data_set1
    input_data = training_data["input_data"]
    true_pos = training_data["true_pos"]
    true_dir = training_data["true_dir"]
    true_kappa = training_data["true_kappa"]
    # nominal_shear = training_data['true_shear']
    input_size = training_data["input_size"]
    output_size = training_data["output_size"]

    E = 10**6
    G = E * 2 / 3
    A = np.pi * (radius.mean(axis=0)) ** 2
    bend_twist_stiff = ((_aver(A)) ** 2 / (4 * np.pi))[None, None, :] * np.diag(
        [E, E, 2 * G]
    )[..., None]

    power_chi_r = 5  # 6 # 5 # 4 # 3
    power_chi_d = 5
    chi_r = 10**power_chi_r  # 1
    chi_d = 10**power_chi_d
    chi_u = 0  # 1e-5

    tensor_constants = TensorConstants(
        bend_twist_stiff,
        idx_data_pts,
        dl,
        chi_r,
        chi_d,
        pca,
        input_size,
        output_size,
    )
    ## Train the model
    num_epochs = int(10)
    batch_size = 128  # 128 # 100
    print(
        "# total samples:",
        len(input_data),
        "# epochs:",
        num_epochs,
        "batch size:",
        batch_size,
    )
    model = CurvatureSmoothing3DModel(
        tensor_constants,
        input_data,
        num_epochs,
        batch_size=batch_size,
        labels=true_kappa,
    )

    model.model_train()

    flag_save = False
    model_name = "data_smoothing_model_br2_BS128"

    if flag_save:
        model.model_save(model_name)


if __name__ == "__main__":

    # main(sys.argv[1])
    main()
