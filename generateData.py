import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

from dataLoader.objects import SimuObject

from utils.util import *
from utils.radon import *
from utils.visualize import *

np.random.seed(1234)
refine = 128
sampleLocation = generateLocation(64)
radonOp = RadonOperator(imageShape=32, refine=refine)

dataFolder = "/export/project/zhixin.sun/MathImgSci/data"
if os.path.exists(dataFolder):
    shutil.rmtree(dataFolder)
os.mkdir(dataFolder)
imgFolder = "/export/project/zhixin.sun/MathImgSci/images"

noiseLevels = {"Low_Noise": 50000,
              "Medium_Noise": 25000,
              "High_Noise": 10000,
              "Very_High_Noise": 2500}

kinds = ["SignalAbsent", "SignalPresent"]

SignalPresent = {"Low_Noise": [],
                 "Medium_Noise": [],
                 "High_Noise": [],
                 "Very_High_Noise": []
                 }

SignalAbsent = {"Low_Noise": [],
                "Medium_Noise": [],
                "High_Noise": [],
                "Very_High_Noise": []
                }
for i in tqdm(range(500)):
    a = SimuObject(10)
    noiseFree_Pre_obj = descretImage(a.getSignalPresentObject(sampleLocation), 32)
    noiseFree_Abs_obj = descretImage(a.getSignalAbsentObject(sampleLocation), 32)
    noiseFree_Pre_sino = radonOp.radon(noiseFree_Pre_obj)
    noiseFree_Abs_sino = radonOp.radon(noiseFree_Abs_obj)
    for nslvl in noiseLevels.keys():
        imgAbs = radonOp.iradon(addGammaToSino(noiseFree_Abs_sino, noiseLevels[nslvl]))
        SignalAbsent[nslvl].append(imgAbs)
        imgPre = radonOp.iradon(addGammaToSino(noiseFree_Pre_sino, noiseLevels[nslvl]))
        SignalPresent[nslvl].append(imgPre)
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(noiseFree_Pre_obj)
        # ax[0, 1].imshow(imgPre)
        # ax[1, 0].imshow(noiseFree_Abs_obj)
        # ax[1, 1].imshow(imgAbs)

    # plt.show()

for nslvl in ["Medium_Noise", "High_Noise", "Very_High_Noise"]:
    with h5py.File(os.path.join(dataFolder, f"SignalAbsent_{nslvl}.hdf5"), "w") as f:
        f.create_dataset("gdt", data=np.stack(SignalAbsent["Low_Noise"], axis=0))
        f.create_dataset("noisy", data=np.stack(SignalAbsent[nslvl], axis=0))
    with h5py.File(os.path.join(dataFolder, f"SignalPresent_{nslvl}.hdf5"), "w") as f:
        f.create_dataset("gdt", data=np.stack(SignalPresent["Low_Noise"], axis=0))
        f.create_dataset("noisy", data=np.stack(SignalPresent[nslvl], axis=0))
