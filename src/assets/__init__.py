ASSETS = "assets"

FILE_NAME_BR2 = {
    4: "BR2_arm_data.npy",
}
MODEL_NAME_BR2 = {
    4: "data_smoothing_model_br2.pt",
}

FILE_NAME_OCTOPUS = {
    0: "octopus_arm_data.npy",
}
for i in range(3,9):
    FILE_NAME_OCTOPUS[i] = "octopus_arm_data_%dbasis.npy"%i

MODEL_NAME_OCTOPUS = {
    0: "data_smoothing_model_octopus.pt"
}
FILE_NAME_OCTOPUS_H5 = "octopus_arm_data.h5"
