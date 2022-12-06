import os
import shutil
import re

import h5py
import numpy as np
import torch



def rect(x):
    return np.all((x <= 0.5), axis=-1).astype(x.dtype)


def circ(r, c, sigma):
    """
    :Param r: a point in R2
    :Param c: a point in R2, the center of the circle
    :Param sigma: the radio of the circle
    """
    return (np.linalg.norm(r - c, ord=2, axis=-1) < sigma).astype(r.dtype)


def unitDisk(x):
    """
    If the norm of x falls in a unit disk, return 1, otherwise 0
    """
    return (np.linalg.norm(x, ord=2, axis=-1) <= 0.5).astype(x.dtype)


def generateLocation(number):
    x = np.linspace(-1/2 + 1 / (number * 2), 1/2 - 1 / (number * 2), number)
    y = np.linspace(-1/2 + 1 / (number * 2), 1/2 - 1 / (number * 2), number)
    gx, gy = np.meshgrid(x, y, indexing="xy")
    coor = np.stack([gx, gy], axis=-1)
    return coor

def generateFileListInFolder(folderPath, keywords=None, extension=None, omitHidden=True, absPath=True, containsDir=False):
    """
    generate all the direct file(and subdirectors) under the given folderPath into a list.
    Args:
        folderPath: The abs path of the folder that we want to list the content of
        extension: just list all the file with the extension if extension is not None
        omitHidden: whether omit hidden files
        absPath: whether the items in the list contains the absPath of the file or just a list of filenames
        containsDir: whether the list contains subdirectors or just files
    Returns:
        a list of the content of the folder
    """
    fileList = []
    for f in os.listdir(folderPath):
        if omitHidden and f.startswith("."):
            continue
        if not containsDir and os.path.isdir(os.path.join(folderPath, f)):
            continue
        if extension is not None:
            if f.endswith(extension):
                fileList.append(f)
            else:
                continue
        else:
            fileList.append(f)
    if keywords is not None:
        keywords = r"\b" + keywords + r"\b"
        keywordsList = []
        for file in fileList:
            if re.search(keywords, file):
                keywordsList.append(file)
        fileList = keywordsList

    if absPath:
        fileList = [os.path.join(folderPath, f) for f in fileList]

    return fileList


def sortFun(x):
    return int(((x.split("/")[-1]).split("-")[-1]).split(".")[0])


def descretImage(img, targetSize):
    """
    reduce the img to the targetSize by averaging the block.
    """
    n = int(img.shape[0] / targetSize)
    # row
    rows = img[::n, :]
    for i in range(1, n):
        rows += img[i::n, :]
    # cols
    cols = rows[:, ::n]
    for i in range(1, n):
        cols += rows[:, i::n]
    return cols / n**2


def addGammaToSino(sinogram, imageScale):
    scale = imageScale / np.sum(sinogram)
    sinogram = sinogram * scale
    return np.random.poisson(sinogram) / scale


# Fundamental Func
evaluateSnr = lambda x, xhat: 20 * np.log10(
    np.linalg.norm(x.flatten('F')) / np.linalg.norm(x.flatten('F') - xhat.flatten('F')))

def compare_snr(img_test, img_true):
    return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))


def copytree(src=None, dst=None, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if (ignore is not None) and (item in ignore):
            print(item)
            continue
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            print("copy to", d)
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


def data_augmentation(image, mode):
    """ flipping or rotating the image
    Args:
    image: 2D matrix,
    mode: int between [0, 7] for different mode
    Returns:
    out: 2D matrix, the image that has been rotated or flipped
    """
    out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return out


# read in files
def read_hdf5_file(file_path, keys: [str] = None, dtype: np.dtype = None):
    data = {}
    with h5py.File(file_path, "r") as f:
        if keys is None:
            keys = list(f.keys())
        for key in keys:
            data[key] = np.array(f[key], dtype=f[key].dtype)
            if dtype is not None:
                data[key] = data[key].astype(dtype)

    return data




