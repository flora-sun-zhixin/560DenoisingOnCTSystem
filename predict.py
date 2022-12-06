# Flora Sun, CIG
import json
import os.path

import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
import torchvision.utils as utils
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.util import *
from utils.Metrics import *
from dataLoader.CustomizeDataset import CustomizeDataset
from models.dncnn import DnCNN

###############################################
#            set config and device            #
###############################################
with open("config.json") as File:
    config = json.load(File)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config["setting"]["gpu_index"]
device = torch.device("cuda:0")
if not torch.cuda.is_available():
    print(device, " is not available.")
    exit

# %%
################## set seed ####################
# %%
np.random.seed(1234)
torch.manual_seed(40)
# %%

###############################################
#   Start Building Dataset and DataLoader     #
###############################################
# %%
torch.cuda.empty_cache()
SignalPresentData = read_hdf5_file(os.path.join(config["dataset"]["data_path"],
                                                f"SignalPresent_{config['dataset']['noiseLevel']}.hdf5"))
SignalAbsentData = read_hdf5_file(os.path.join(config["dataset"]["data_path"],
                                               f"SignalAbsent_{config['dataset']['noiseLevel']}.hdf5"))
trainIndex = np.random.choice(500, 300, replace=False)
restIndex = np.setdiff1d(np.arange(500, dtype=trainIndex.dtype), trainIndex)
validIndex = np.random.choice(restIndex, 100, replace=False)
testIndex = np.setdiff1d(restIndex, validIndex)

testDict = {"gdt": np.concatenate([SignalAbsentData["gdt"][testIndex, ...],
                                   SignalPresentData["gdt"][testIndex, ...]], axis=0),
            "noisy": np.concatenate([SignalAbsentData["noisy"][testIndex, ...],
                                     SignalPresentData["noisy"][testIndex, ...]], axis=0)}
testDataset = CustomizeDataset(testDict, shuffle=False)
testLoader = DataLoader(testDataset, batch_size=config["test"]["batch_size"], shuffle=False)

# %%
###############################################
#       End Building Dataset and DataLoader   #
###############################################

###############################################
#                 Start Training              #
###############################################

# set path
print('Begin prediction !')
init_lr = config['train']['init_lr']
global_step = 0
dnn = DnCNN(depth=config["cnn_model"]["depth"], n_channels=config["cnn_model"]["n_channels"],
            image_channels=config["cnn_model"]["image_channels"], kernel_size=config["cnn_model"]["kernel_size"])
# dnn.apply(weights_init_kaiming)
network = dnn.to(device)

# continue training
model_path = os.path.join(config["root_path"],
                          "experiements",
                          config["test"]["model_path"])
model_file = os.path.join(model_path, "logs", config["test"]["model_file"])
checkpoint = torch.load(model_file)
network.load_state_dict(checkpoint["model_state_dict"])

loss_fn_l1 = nn.L1Loss().to(device)

optimizer = optim.Adam([{"params": network.parameters()}], lr=init_lr, weight_decay=config["train"]["weight_decay"])

############################################
#                   Test                   #
############################################
# set training = False
network.eval()
test_pred = []
avg_snr_val = 0
avg_psnr_val = 0
avg_ssim_val = 0
avg_loss_val = 0
batch_val = 0
for batch_data in tqdm(testLoader):
    batch_data = batch_data.to(device)
    batch_gdt = batch_data[:, :1]
    batch_ipt = batch_data[:, 1:]
    batch_size = batch_gdt.shape[0]
    batch_pre = network(batch_ipt)
    test_pred.append(batch_pre)
    snr = compare_snr_batch(batch_pre, batch_gdt) * batch_size
    psnr = compare_psnr_batch(batch_pre, batch_gdt) * batch_size
    ssim = compare_ssim_batch(batch_pre, batch_gdt) * batch_size
    avg_snr_val = avg_snr_val + snr
    avg_psnr_val = avg_psnr_val + psnr
    avg_ssim_val = avg_ssim_val + ssim
    loss = loss_fn_l1(batch_pre, batch_gdt)
    avg_loss_val = avg_loss_val + loss.item()
    batch_val = batch_val + batch_size
avg_snr_val = avg_snr_val / batch_val
avg_psnr_val = avg_psnr_val / batch_val
avg_ssim_val = avg_ssim_val / batch_val
avg_loss_val = avg_loss_val / batch_val
with torch.no_grad():
    print("==================================\n")
    print('batch_pre: ', avg_loss_val)
    print('snr: ', avg_snr_val, "psnr: ", avg_psnr_val, "ssim: ", avg_ssim_val)
    print("\n==================================")
    print("")

predData = torch.cat(test_pred, dim=0).double().squeeze().data.cpu().numpy()
result_path = os.path.join(model_path, "results")
if not os.path.exists(result_path):
    os.mkdir(result_path)

labelData = np.concatenate([np.zeros((predData.shape[0] // 2,)), np.ones((predData.shape[0] // 2,))], axis=0)
for i in list(range(50)):# + list(range(100, 150)):
    fig, axes = plt.subplots(1, 3)
    plt.tight_layout()
    axes[0].imshow(testDict["gdt"][i, ...])
    axes[0].set_title("Ground Truth", fontsize=40)
    axes[0].axis("off")
    axes[1].imshow(testDict["noisy"][i, ...])
    axes[1].set_title(config["dataset"]["noiseLevel"], fontsize=40)
    axes[1].axis("off")
    axes[2].imshow(predData[i, ...])
    axes[2].set_title("Denoised Result", fontsize=40)
    axes[2].axis("off")
    # fig.savefig(os.path.join(result_path, f'result-{config["dataset"]["noiseLevel"]}'), bbox_inches="tight")

plt.show()
print(predData.shape)
print(testDict["gdt"].shape)

with h5py.File(os.path.join(result_path, f"pred_{config['dataset']['noiseLevel']}.hdf5"), "w") as f:
    f.create_dataset("predict", data=predData)
    f.create_dataset("gdt", data=testDict["gdt"])
    f.create_dataset("noisy", data=testDict["noisy"])
    f.create_dataset("label", data=labelData)