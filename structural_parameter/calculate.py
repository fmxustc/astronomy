import image_ops
import warnings
import numpy as np
import pandas as pd


def fi(fn):
    return path + 'image/' + fn + '_r.fits'


def fs(fn):
    return path + 'seg/' + fn + '_r.fits'


def cal(gal, tp):

    gn = gal.NAME1 if tp == 1 else gal.NAME2
    center = [gal.RA1, gal.DEC1] if tp == 1 else [gal.RA2, gal.DEC2]

    init, seg = image_ops.load(fi(gn), fs(gn), center[0], center[1])
    resc = image_ops.rescale(init, log_scale=True, method=adpt)

    seg180 = np.rot90(seg, 2)
    flag = seg[seg.shape[0]/2][seg.shape[1]/2]
    zero = np.zeros(seg.shape)
    # TODO: bugs
    I = np.where(((abs(seg-flag) > 1) & seg) | ((abs(seg180-flag) > 1) & seg180) | (seg == 0), zero, resc)
    I180 = np.rot90(I, 2)
    print(np.sum(np.abs(I-I180))/np.sum(np.abs(I)))
    image_ops.show(seg)
    return


warnings.filterwarnings('ignore')
path = '/home/franky/Desktop/type1/'
name = 'J083732.70+284218.7'
adpt = 'adaptive_equalization'
sigmoid = 'adjust_sigmoid'
gleq = 'global_equalization'
loeq = 'local_equalization'

catalog = pd.read_csv('list.csv')
sample = catalog.ix[52]
cal(sample, 1)

