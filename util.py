"""Utility

Author: Ruangrawee (Kao) Kitichotkul
"""

import numpy as np
from PIL import Image
from skimage import metrics as skm

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import os
from pathlib import Path
import json

"""Evaluation metrics"""

def computeMSE(x1, x2):
    return np.mean((x1 - x2) ** 2)

def computeRMSE(x1, x2):
    return np.sqrt(computeMSE(x1, x2))

def computePSNR(x1, x2, dynamicRange=1.0):
    return 20 * np.log10(dynamicRange / computeRMSE(x1, x2))

def computePSNRfromMSE(mse, dynamicRange):
    return 20 * np.log10(dynamicRange / np.sqrt(mse))

def computeSNR(testImage, gtImage):
    """Compute SNR without assuming noise model
    
    SNR (in dB) = 10 log_10 (E[signal^2] / E[noise^2])
    """
    noise = testImage - gtImage
    noisePower = np.mean(noise ** 2)
    signalPower = np.mean(testImage ** 2)
    return 10 * np.log10(signalPower / noisePower)

def computeErrorMetrics(testImage, gtImage, dynamicRange):
    mse = computeMSE(testImage, gtImage)
    rmse = np.sqrt(mse)
    psnr = computePSNRfromMSE(mse, dynamicRange)
    snr = computeSNR(testImage, gtImage)
    ssim = skm.structural_similarity(testImage, gtImage, data_range=dynamicRange)
    return mse, rmse, psnr, snr, ssim

def showErrorMetrics(testImage, gtImage, dynamicRange, text = None):
    mse, rmse, psnr, snr, ssim = computeErrorMetrics(testImage, gtImage, dynamicRange)
    if text is not None:
        print(text)
    print('MSE = {}'.format(mse))
    print('RMSE = {}'.format(rmse))
    print('PSNR = {}'.format(psnr))
    print('SNR = {}'.format(snr))
    print('SSIM = {}'.format(ssim))

"""Plotting"""

def imshow(images, numCols=3, vmin=None, vmax=None, savename=None):
    numImages = len(images)
    if isinstance(images, list):
        numCols = np.minimum(numImages, numCols)
        numRows = int(np.ceil(numImages / numCols))
        fig, ax = plt.subplots(numRows, numCols, figsize=(10, 5))
        ax = ax.reshape(-1)
        for i in range(numImages):
            if (vmin is None and vmax is None) or i == numImages - 1:
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                im = ax[i].imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax)
                fig.colorbar(im, cax=cax)
            else:
                ax[i].imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax)
            ax[i].axis('off')
        for i in range(numImages, numCols * numRows):
            fig.delaxes(ax[i])
        fig.tight_layout()
    else: # assume one image
        plt.imshow(images, cmap='gray', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.axis('off')

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename, bbox_inches='tight')
        plt.clf()
        plt.close()

def summarizeSolve(hist):
    plt.plot(hist['mse'])
    plt.xlabel('iteration')
    plt.title('MSE')
    plt.show()

def plot3d(x, y, z, labels):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.1f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10, location='right', pad=0.1)

    ax.set_xlabel(labels['x'])
    ax.set_ylabel(labels['y'])
    ax.set_zlabel(labels['z'])

    plt.show()

"""I/O"""

def readImage(path):
    return np.array(Image.open(path).convert('L')).astype(float) / 255.

def readImageToEta(path, minEta, maxEta):
    assert minEta <= maxEta
    image = readImage(path)
    image = image * (maxEta - minEta) + minEta
    return image

def saveImage(image, minVal, maxVal, path):
    clippedImage = np.clip(image, minVal, maxVal)
    rescaledImage = np.uint8((clippedImage - minVal) / (maxVal - minVal) * 255)
    im = Image.fromarray(rescaledImage)
    im.save(path)

def logArguments(args, path):
    """Write arguments in a given argparse.Namespace object to a text file."""
    with open(path, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

def saveDict(d, filename):
    with open(filename, 'w') as f:
        json.dump(d, f)

def loadDict(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def prepareImagePath(datadir, numImages = None):
    """Generate loadlist and namelist"""
    loadlist = generateLoadlist(datadir, num_files=numImages)
    namelist = generateNamelist(datadir, num_files=numImages, no_exten=True)
    numImages = len(loadlist)
    return loadlist, namelist, numImages

def generateLoadlist(datadir, prefix=None, suffix=None, num_files=None):
    """Generate list of paths to images"""
    namelist = generateNamelist(datadir, prefix=prefix, suffix=suffix, num_files=num_files)
    if num_files is None or len(namelist) < num_files:
        num_files = len(namelist)
    loadlist = [None] * num_files
    for i, name in enumerate(namelist):
        loadlist[i] = os.path.join(datadir, name)
    return loadlist

def generateNamelist(datadir, num_files=None, prefix=None, suffix=None, no_exten=False, no_hidden=True):
    """Generate list of file names in a directory
    
    Args:
        datadir (std): path to directory containing files.
        num_files (int): number of files to read. If a number is given, return first num_files names by 
            lexicographical order. If None, read all files satisfying other criteria (prefix, etc.).
        prefix (str): return only file names beginning with this prefix.
        suffix (str): return only file names (including the extension) ending with this suffix.
        no_exten (bool): whether to include the extension in the returning file names.
        no_hidden (bool): whether to include hidden files.
    Returns:
        namelist (list): list of file names.
    """
    raw_list = sorted(os.listdir(datadir))
    if prefix is not None:
        prefix_filtered_list = []
        for name in raw_list:
            if name.startswith(prefix):
                prefix_filtered_list.append(name)
    else:
        prefix_filtered_list = raw_list
    if suffix is not None:
        filtered_list = []
        for name in prefix_filtered_list:
            if name.endswith(suffix):
                filtered_list.append(name)
    else:
        filtered_list = prefix_filtered_list

    if num_files is None or len(filtered_list) < num_files:
        num_files = len(filtered_list)

    if no_exten:
        namelist = [None] * num_files
        namelist_with_exten = filtered_list[:num_files]
        for i, name in enumerate(namelist_with_exten):
            namelist[i] = os.path.splitext(name)[0]
    else:
        namelist = filtered_list[:num_files]
    
    if no_hidden:
        namelist = [name for name in namelist if not name.startswith('.')]

    return namelist