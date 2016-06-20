import image_ops
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank


def plot_img_and_hist(img, axes, bins=512):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap='Greys')
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[img.dtype.type]
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf


warnings.filterwarnings('ignore')
path = '/Users/franky/Desktop/type1/'
name = 'J083732.70+284218.7'
init, seg = image_ops.load(path+'image/'+name+'_r.fits', path+'seg/'+name+'_r.fits', 129.38627,28.705196)
# resc = image_ops.contrast_stretching(init, 2, 98)
contr = image_ops.contrast_stretching(init, 5, 99)
adpat = image_ops.rescale(init, method='adaptive_equalization', log_scale=True)
adjust = image_ops.rescale(init, method='adjust_sigmoid', log_scale=True)
loca = image_ops.rescale(init, method='global_equalization', log_scale=True)
glob = image_ops.rescale(init, method='local_equalization', log_scale=True)


def show():
    # Display results
    fig = plt.figure(figsize=(14, 10))
    axes = np.zeros((2, 6), dtype=np.object)
    axes[0, 0] = plt.subplot(2, 6, 1, adjustable='box-forced')
    axes[0, 1] = plt.subplot(2, 6, 2, sharex=axes[0, 0], sharey=axes[0, 0],
                             adjustable='box-forced')
    axes[0, 2] = plt.subplot(2, 6, 3, sharex=axes[0, 0], sharey=axes[0, 0],
                             adjustable='box-forced')
    axes[0, 3] = plt.subplot(2, 6, 4, sharex=axes[0, 0], sharey=axes[0, 0],
                             adjustable='box-forced')
    axes[0, 4] = plt.subplot(2, 6, 5, sharex=axes[0, 0], sharey=axes[0, 0],
                             adjustable='box-forced')
    axes[0, 5] = plt.subplot(2, 6, 6, sharex=axes[0, 0], sharey=axes[0, 0],
                             adjustable='box-forced')
    axes[1, 0] = plt.subplot(2, 6, 7)
    axes[1, 1] = plt.subplot(2, 6, 8)
    axes[1, 2] = plt.subplot(2, 6, 9)
    axes[1, 3] = plt.subplot(2, 6, 10)
    axes[1, 4] = plt.subplot(2, 6, 11)
    axes[1, 5] = plt.subplot(2, 6, 12)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(init, axes[:, 0])
    ax_img.set_title('Low contrast image')
    ax_hist.set_ylabel('Number of pixels')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(contr, axes[:, 1])
    ax_img.set_title('Contrast stretched')
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(adpat, axes[:, 2])
    ax_img.set_title('Adaptive Equalization')
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(adjust, axes[:, 3])
    ax_img.set_title('Adjust Sigmoid')
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(glob, axes[:, 4])
    ax_img.set_title('Global Equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(loca, axes[:, 5])
    ax_img.set_title('local Equalization')
    ax_cdf.set_ylabel('Fraction of total intensity')

    fig.tight_layout()
    plt.show()


print(adjust.shape, init.shape)
show()
