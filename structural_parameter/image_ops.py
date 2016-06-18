import astropy.io.fits as ft
import astropy.wcs as wcs
import numpy as np
import scipy.misc as misc
import skimage.exposure as exposure
import skimage.filters.rank as rank
from skimage.morphology import disk
import skimage.data as skd


"""Image Operations for loading, showing and rescaling fits file"""


def load(luminous_file, seg_file, ra, dec):
    hdu = ft.open(luminous_file)
    header = hdu[0].header
    initial_data = hdu[0].data
    seg_data = ft.open(seg_file)[0].data
    cy, cx = np.round(wcs.WCS(header).wcs_world2pix(ra, dec, 0))
    y, x = np.where(seg_data == seg_data[cy][cx])
    flux = initial_data[y, x]
    arg = np.argsort(flux)[::-1]
    cumulative_flux = flux[arg]
    for i in range(1, len(flux)):
        cumulative_flux[i] += cumulative_flux[i-1]
    qpr = int(np.argwhere(flux[arg]-0.2*cumulative_flux/np.arange(1, len(flux)+1))[0])
    radius = np.max(np.sqrt((y-cy)**2+(x-cx)**2))*0.9
    return initial_data[cy-radius:cy+radius+1, cx-radius:cx+radius+1], seg_data[cy-radius:cy+radius+1, cx-radius:cx+radius+1]


def show(img_data):
    misc.imshow(-img_data)
    return


def rescale(initial_data, log_scale=True, method='adaptive_equalization'):
    norm = (initial_data-initial_data.min())/initial_data.max()
    exponent = 1000
    log = np.log(exponent*norm+1)/np.log(exponent) if log_scale else norm
    if method == 'adaptive_equalization':
        return exposure.equalize_adapthist(log/log.max(), nbins=2048, kernel_size=128)
    elif method == 'adjust_sigmoid':
        return exposure.adjust_sigmoid(log/log.max(), cutoff=0.5, gain=20)
    elif method == 'global_equalization':
        return exposure.equalize_hist(log / log.max(), nbins=1024)
    elif method == 'local_equalization':
        return rank.equalize(log / log.max(), selem=disk(50))
    return


def contrast_stretching(initial_data, lp, rp):
    pl, pr = np.percentile(initial_data, lp), np.percentile(initial_data, rp)
    return exposure.rescale_intensity(initial_data, in_range=(pl, pr))
