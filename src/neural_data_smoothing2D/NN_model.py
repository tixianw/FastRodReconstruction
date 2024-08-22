"""
Created on Jul 17, 2024
@author: Tixian Wang
"""

import sys

import numpy as np
import numpy.random as npr
import torch
import torch.nn.functional as F
from run_PCA import PCA
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from utils import coeff2curvature_torch, coeff2position_torch

# from numpy.linalg import eig
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# torch.manual_seed(2024)
# npr.seed(2024)


class TensorConstants:
    def __init__(self, EI, idx_data_pts, ds, chi_r, chi_u, pca):
        self.EI = torch.from_numpy(EI).float()
        self.idx_data_pts = torch.from_numpy(idx_data_pts)
        self.ds = ds
        self.chi_r = chi_r
        self.chi_u = chi_u
        self.pca_components = torch.from_numpy(pca.components).float()
        self.pca_mean = torch.from_numpy(pca.mean).float()
        self.pca_std = torch.from_numpy(pca.std).float()


class CustomLoss(nn.Module):
    def __init__(self, tensor_constants):
        super().__init__()
        self.tensor_constants = tensor_constants

    def forward(self, outputs, labels):
        # outputs = torch.flatten(outputs)
        # labels = torch.from_numpy(labels).float()
        # kappa_hat = torch.mv(self.tensor_constants.pca_components, outputs)
        # kappa_hat = torch.einsum('nj,ij->ni', outputs, self.tensor_constants.pca_components)
        kappa_hat = coeff2curvature_torch(outputs, self.tensor_constants)
        diff = kappa_hat - labels
        loss = (
            0.5 * torch.sum(diff * diff, axis=1) * self.tensor_constants.ds
        )  # self.tensor_constants.EI *
        return loss.mean()


class CurvatureSmoothing2DLoss(nn.Module):
    def __init__(self, tensor_constants):
        super().__init__()
        self.tensor_constants = tensor_constants

    def potential_energy(self):
        V = (
            0.5
            * torch.sum(
                self.tensor_constants.EI * self.kappa_hat * self.kappa_hat,
                axis=1,
            )
            * self.tensor_constants.ds
        )
        return V.mean()

    def derivative_penalty(self, outputs):
        kappa_hat_s_scaled = torch.einsum(
            "nj,ij->ni", outputs[..., 1:], self.tensor_constants.pca_components
        )  ## n is batch size
        kappa_hat_s = (
            kappa_hat_s_scaled * self.tensor_constants.pca_std
            + self.tensor_constants.pca_mean
        )
        E_u = (
            0.5
            * self.tensor_constants.chi_u
            * torch.sum(kappa_hat_s * kappa_hat_s)
            * self.tensor_constants.ds
        )
        return E_u.mean()

    def data_matching_cost(self, inputs, outputs):
        inputs = torch.flatten(inputs, start_dim=1)
        position_difference = inputs - coeff2position_torch(
            outputs, self.tensor_constants
        )
        Phi = (
            0.5
            * self.tensor_constants.chi_r
            * torch.sum(position_difference * position_difference, axis=1)
        )
        return Phi.mean()

    def forward(self, outputs, inputs):
        """
        input_data: discrete position data points
        outputs: weights of PCA compunents for approximated target curvature
        """
        # inputs = torch.flatten(inputs)
        # outputs = torch.flatten(outputs)
        # self.kappa_hat = torch.mv(self.tensor_constants.pca_components, outputs)
        self.kappa_hat = coeff2curvature_torch(outputs, self.tensor_constants)
        J = self.potential_energy()
        # print('V:', J.data, 'E:', self.derivative_penalty(outputs).data, 'Phi:', self.data_matching_cost(inputs, outputs).data)
        # J += self.derivative_penalty(outputs)
        J += self.data_matching_cost(inputs, outputs)
        # J = self.data_matching_cost(inputs, outputs)
        return J


class CurvatureSmoothing2DNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = (
            nn.SiLU()
        )  # nn.ReLU() # nn.GELU() # convergence speed: GELU > SiLU > ReLU
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(input_size, 512),
            # self.activation,
            # nn.Linear(512, 256),
            # self.activation,
            # nn.Linear(256, 128),
            # self.activation,
            # nn.Linear(128, 64),
            # self.activation,
            # nn.Linear(64, output_size),
            # nn.Linear(input_size, 32),
            # self.activation,
            # nn.Linear(32, 32),
            # self.activation,
            # nn.Linear(32, 32),
            # self.activation,
            # nn.Linear(32, output_size)
            nn.Linear(input_size, 20),
            self.activation,
            nn.Linear(20, 15),
            self.activation,
            nn.Linear(15, output_size),
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


class CurvatureSmoothing2DModel:
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
        self.input_size = len(tensor_constants.idx_data_pts) * 2
        self.output_size = tensor_constants.pca_components.shape[1]  # + 1
        self.net = CurvatureSmoothing2DNet(self.input_size, self.output_size)
        self.loss_fn = CurvatureSmoothing2DLoss(
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

    def model_train(self):
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

        self.test_loss = 0.0
        for i, test_inputs in enumerate(self.test_loader):
            test_outputs = self.net(test_inputs)
            t_loss = self.loss_fn(test_outputs, test_inputs)
            self.test_loss += t_loss.item()
        self.test_loss /= i + 1
        print(f"test loss: {self.test_loss:.8f}")

    def model_save(self, model_name):
        torch.save(
            {
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "tensor_constants": self.tensor_constants,
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "losses": [
                    self.train_losses,
                    self.validation_losses,
                    self.test_loss,
                ],
            },
            "Data/model/" + model_name + ".pt",
        )


def main():
    # print('Cuda is available?', torch.cuda.is_available())
    # # torch.cuda.device_count()
    # # torch.cuda.current_device()

    folder_name = "Data/"
    file_name = "fake_br2_arm_data1"  # 'pyelastica_arm_data2' #
    data = np.load(folder_name + file_name + ".npy", allow_pickle="TRUE").item()

    n_elem = data["model"]["n_elem"]
    L = data["model"]["L"]
    radius = data["model"]["radius"]
    s = data["model"]["s"]
    ds = s[1] - s[0]
    idx_data_pts = data["idx_data_pts"]
    input_data = data["input_data"]
    true_pos = data["true_pos"]
    true_kappa = data["true_kappa"]
    pca = data["pca"]

    training_data = np.load(
        folder_name + "training_data_set1.npy", allow_pickle="TRUE"
    ).item()  # training_data_set1
    input_data = training_data["training_data"]
    true_pos = training_data["true_pos"]
    true_kappa = training_data["true_kappa"]

    A = np.pi * (0.5 * (radius[1:] + radius[:-1])) ** 2
    EI = 10**6 * (0.5 * (A[1:] + A[:-1])) ** 2 / (4 * np.pi)
    power_chi_r = 5  # 6 # 5 # 4 # 3
    chi_r = 10**power_chi_r  # 1
    chi_u = 0  # 1e-5

    tensor_constants = TensorConstants(EI, idx_data_pts, ds, chi_r, chi_u, pca)
    ## Train the model
    num_epochs = int(50)
    batch_size = 128  # 100
    print(
        "# total samples:",
        len(input_data),
        "# epochs:",
        num_epochs,
        "batch size:",
        batch_size,
    )
    model = CurvatureSmoothing2DModel(
        tensor_constants,
        input_data,
        num_epochs,
        batch_size=batch_size,
        labels=true_kappa,
    )

    model.model_train()

    flag_save = True
    # # model_name = 'kappa_l2norm_model1' # 'kappa_mse_model1'
    # # model_name = 'data_smoothing_model_chi1e'+str(power_chi_r) # 'small_model_test1'
    model_name = "mini_batch_model_br2"  # pyelastica'
    # model_name = 'curvature_derivative_model_pyelastica' # _chi1e-6_1e6_SiLU'

    if flag_save:
        model.model_save(model_name)


if __name__ == "__main__":

    # main(sys.argv[1])
    main()
