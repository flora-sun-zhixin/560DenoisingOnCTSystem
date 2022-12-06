import matplotlib.pyplot as plt
import numpy as np
import imageio

from utils.util import *


def plot_h_all_detector(H, which_angle: int, total_detector=16, cmap="tab20b", dpi=None):
    pixel_num = H.shape[1]
    n = int(np.sqrt(pixel_num))

    holder = np.zeros((n*n,))
    for which_detector in range(total_detector):
        tmp = H[:, which_angle * total_detector + which_detector].reshape(n, n)
        holder += H[which_angle * total_detector + which_detector, :] * (which_detector + 1)

    holder = np.ma.masked_where(holder==0, holder)
    fig, ax = plt.subplots(1, 1, dpi=dpi)
    holder = holder.reshape(n, n)
    ax.imshow(holder, cmap=cmap, interpolation="None")
    ax.set_title(f"H at {which_angle}-th angle")
    ax.set_aspect("equal", "box")
    return fig


def plot_h_one_detector_one_angle(H, which_angle: int, which_detector: int, total_detector=16, cmap="tab20b", dpi=None):
    pixel_num = H.shape[1]
    n = int(np.sqrt(pixel_num))
    holder = H[which_angle * total_detector + which_detector, :]
    holder = np.ma.masked_where(holder==0, holder)
    holder = holder.reshape(n, n)
    fig = plt.figure(dpi=dpi)
    plt.imshow(holder, cmap=cmap, interpolation="None")
    plt.title(f"H at {which_angle}-th angle and {which_detector}-th detector")
    return fig


def generateMP4_H(imageFolder, mp4name, sortFunction, keywords=None, extension="png"):
    picList = generateFileListInFolder(imageFolder, keywords=keywords, extension=extension)
    writer = imageio.get_writer(os.path.join(imageFolder, mp4name), fps=3)
    imageList = sorted(picList, key=sortFunction)
    for im in imageList:
        writer.append_data(imageio.imread(im))
    writer.close()


def plotImageWithGrid(img, title):
    n = img.shape[0]
    fig = plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.hlines(y=np.arange(0, n) - 0.5, xmin=np.full(n, 0) - 0.5,
               xmax=np.full(n, n) - 0.5, color="black", linewidth=0.2)
    plt.vlines(x=np.arange(0, n) - 0.5, ymin=np.full(n, 0) - 0.5,
               ymax=np.full(n, n) - 0.5, color="black", linewidth=0.2)
    return fig


def plotSinogram(sinogram, title, ttlNumDtctr=16, ttlNumAngle=8):
    sinogram = sinogram.reshape(ttlNumAngle, ttlNumDtctr)
    fig = plt.figure()
    plt.imshow(sinogram)
    plt.title(title)
    plt.axis(False)
    return fig


def plotImage(image, imageShape, title):
    image = image.reshape(imageShape, imageShape)
    fig = plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis(False)
    return fig
