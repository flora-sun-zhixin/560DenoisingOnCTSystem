# Flora Sun, CIG
import json
from datetime import datetime

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
###############################################
#                 Set save path               #
###############################################
# %%
now = datetime.now()
save_root = config['root_path'] + '/experiements/%s' % (str(now.strftime("%d-%b-%Y-%H-%M-%S"))) + \
            "%s_depth_%d_sigma_%s" % (
            config['cnn_model']['network'], config['cnn_model']['depth'], config['dataset']["noiseLevel"])
copytree(src=config['code_path'], dst=save_root, ignore=[".git",".gitignore"])

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
validIndex = np.random.choice(restIndex, 100)
testIndex = np.setdiff1d(restIndex, validIndex)

trainDict = {"gdt": np.concatenate([SignalAbsentData["gdt"][trainIndex, ...],
                                    SignalPresentData["gdt"][trainIndex, ...]], axis=0),
             "noisy": np.concatenate([SignalAbsentData["noisy"][trainIndex, ...],
                                      SignalPresentData["noisy"][trainIndex, ...]], axis=0)}
trainDataset = CustomizeDataset(trainDict)

validDict = {"gdt": np.concatenate([SignalAbsentData["gdt"][validIndex, ...],
                                    SignalPresentData["gdt"][validIndex, ...]], axis=0),
             "noisy": np.concatenate([SignalAbsentData["noisy"][validIndex, ...],
                                      SignalPresentData["noisy"][validIndex, ...]], axis=0)}
validDataset = CustomizeDataset(validDict)

testDict = {"gdt": np.concatenate([SignalAbsentData["gdt"][testIndex, ...],
                                   SignalPresentData["gdt"][testIndex, ...]], axis=0),
            "noisy": np.concatenate([SignalAbsentData["noisy"][testIndex, ...],
                                     SignalPresentData["noisy"][testIndex, ...]], axis=0)}

trainLoader = DataLoader(trainDataset, batch_size=config["train"]["batch_size"], shuffle=True)
validLoader = DataLoader(validDataset, batch_size=config["valid"]["batch_size"], shuffle=True)

data_augment = torch.nn.Sequential(
    T.RandomHorizontalFlip(p=0.3),
    T.RandomVerticalFlip(p=0.3)
)
# %%
###############################################
#       End Building Dataset and DataLoader   #
###############################################

###############################################
#                 Start Training              #
###############################################
# %%
writer = SummaryWriter(log_dir=save_root + "/logs")

# set path
print('Prepare training !')
init_lr = config['train']['init_lr']
global_step = 0
dnn = DnCNN(depth=config["cnn_model"]["depth"], n_channels=config["cnn_model"]["n_channels"],
            image_channels=config["cnn_model"]["image_channels"], kernel_size=config["cnn_model"]["kernel_size"])
# dnn.apply(weights_init_kaiming)
network = dnn.to(device)

# continue training
# model_path = os.path.join(config["root_path"],
#                           "Experiements/potential_dncnn",
#                           config["test"]["model_path"])
# model_file = os.path.join(model_path, "logs", config["test"]["model_file"])
# checkpoint = torch.load(model_file)
# network.load_state_dict(checkpoint["model_state_dict"])

loss_fn_l1 = nn.L1Loss().to(device)

best_snr = 0

optimizer = optim.Adam([{"params": network.parameters()}], lr=init_lr, weight_decay=config["train"]["weight_decay"])

for epoch in range(1, config['train']['end2end_epoch']):
    # set training = True
    network.train()
    # set learning rate
    # in the first 60 epoch, use fix step size
    if epoch < config['train']['end2end_milestone']:
        current_lr = config['train']['end2end_lr']
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
    # after the first 60 epoch, the step size shrinks exponentially and steply per 25 epoches.
    elif epoch > config['train']['end2end_milestone'] and epoch % 25 == 0:
        current_lr = current_lr / 1.1
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
    print('learning rate %f' % current_lr)

    for batch_data in tqdm(trainLoader):
        batch_data = batch_data.to(device)
        batch_data = data_augment(batch_data)
        batch_gdt, batch_ipt = batch_data[:, :1], batch_data[:, 1:]

        optimizer.zero_grad()
        # batch_ipt.requires_grad_()
        batch_pre = network(batch_ipt)

        loss = loss_fn_l1(batch_pre, batch_gdt)

        snr = compare_snr_batch(batch_pre, batch_gdt)
        psnr = compare_psnr_batch(batch_pre, batch_gdt)
        ssim = compare_ssim_batch(batch_pre, batch_gdt)
        writer.add_scalar('train/snr_iter', snr, global_step)
        writer.add_scalar('train/psnr_iter', psnr, global_step)
        writer.add_scalar('train/ssim_iter', ssim, global_step)
        writer.add_scalar('train/loss_iter', loss, global_step)
        loss.backward()
        optimizer.step()

        global_step += 1

    with torch.no_grad():
        Img_pre = utils.make_grid(batch_pre, nrow=4, normalize=True, scale_each=True)
        Img_gdt = utils.make_grid(batch_gdt, nrow=4, normalize=True, scale_each=True)
        writer.add_scalar('train/lr', current_lr, epoch)
        writer.add_image('train/pre', Img_pre, epoch, dataformats='CHW')
        writer.add_image('train/gt', Img_gdt, epoch, dataformats='CHW')
        writer.add_histogram('train/histogrsm', batch_pre, epoch)

    writer.flush()
    ############################################
    #                   Valid                  #
    ############################################
    # set training = False
    network.eval()
    avg_snr_val = 0
    avg_psnr_val = 0
    avg_ssim_val = 0
    avg_loss_val = 0
    batch_val = 0
    for batch_data in tqdm(validLoader):
        batch_data = batch_data.to(device)
        batch_gdt = batch_data[:, :1]
        batch_ipt = batch_data[:, 1:]
        batch_size = batch_gdt.shape[0]
        batch_pre = network(batch_ipt)
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
        writer.add_scalar('valid/snr', avg_snr_val, epoch)
        writer.add_scalar('valid/psnr', avg_psnr_val, epoch)
        writer.add_scalar('valid/ssim', avg_ssim_val, epoch)
        writer.add_scalar('valid/loss', avg_loss_val, epoch)
        writer.add_histogram('valid/histogrsm', batch_pre, epoch)
        print("==================================\n")
        print("epoch: [%d]" % epoch)
        print('batch_pre: ', torch.std(batch_pre))
        print('snr: ', avg_snr_val, "psnr: ", avg_psnr_val, "ssim: ", avg_ssim_val)
        print("\n==================================")
        print("")
        Img_pre = utils.make_grid(batch_pre, nrow=3, normalize=True, scale_each=True)
        Img_ipt = utils.make_grid(batch_ipt, nrow=3, normalize=True, scale_each=True)
        Img_gdt = utils.make_grid(batch_gdt, nrow=3, normalize=True, scale_each=True)
        writer.add_image('valid/denoise', Img_pre, epoch, dataformats='CHW')
        writer.add_image('valid/ipt', Img_ipt, epoch, dataformats='CHW')
        writer.add_image('valid/gt', Img_gdt, epoch, dataformats='CHW')

    writer.flush()
    if epoch % config['train']['save_epoch'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
        }, save_root + '/logs/dncnn_scalar_image_epoch_mid.pth')

    if avg_loss_val > best_snr:
        best_snr = avg_loss_val
        torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
        }, save_root + '/logs/dncnn_scalar_image_best_valid.pth')

torch.save({
    'epoch': epoch,
    'model_state_dict': network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_l2': loss_fn_l1,
}, save_root + '/logs/dncnn_scalar_image_final.pth')

writer.close()
###############################################
#                  End Training               #
###############################################

