import image_ops
import warnings
import numpy as np
import pandas as pd


def fi(fn):
    return path + 'image/' + fn + '_r.fits'


def fs(fn):
    return path + 'seg/' + fn + '_r.fits'


def fc(fn):
    return path + 'catalog/' + fn + '_r.txt'


def cal(gal, tp):

    gn = gal.NAME1 if tp == 1 else gal.NAME2
    print(gn)
    center = [gal.RA1, gal.DEC1] if tp == 1 else [gal.RA2, gal.DEC2]

    init, seg = image_ops.load(fi(gn), fs(gn), center[0], center[1])
    ctl = pd.read_csv(fc(gn), names=['mag', 'x', 'y', 'fi', 'fp'], sep='\s+')
    adp = image_ops.rescale(init, log_scale=True, method=adpt)
    sig = image_ops.rescale(init, log_scale=True, method=sigmoid)
    geq = image_ops.rescale(init, log_scale=True, method=gleq)
    leq = image_ops.rescale(init, log_scale=True, method=loeq)
    con = image_ops.contrast_stretching(init, 5, 95)

    seg180 = np.rot90(seg, 2)
    flag = seg[seg.shape[0]/2][seg.shape[1]/2]
    zero = np.zeros(seg.shape)
    def asym(arr):
        I = np.where((abs(seg-flag) > 2 & seg) | (abs(seg180-flag) > 2 & seg180), zero, arr)
        I180 = np.rot90(I, 2)
        return np.sum(np.abs(I-I180))/np.sum(np.abs(I))
    # print(ctl.ix[[0,111]])
    # print(type(ctl.ix[flag].fi))
    image_ops.show(np.where(float(ctl.ix[seg].fi) > 2, zero, adp))
    # image_ops.show(np.where((abs(seg-flag) > 2 & seg) | (abs(seg180-flag) > 2 & seg180) , zero, adp))
    # image_ops.show(adp)
    return asym(init), asym(con), asym(adp), asym(sig), asym(geq), asym(leq)


warnings.filterwarnings('ignore')
path = '/home/franky/Desktop/type1/'
name = 'J083732.70+284218.7'
adpt = 'adaptive_equalization'
sigmoid = 'adjust_sigmoid'
gleq = 'global_equalization'
loeq = 'local_equalization'

catalog = pd.read_csv('data.csv', sep=' ')
# print(catalog)
sample = catalog.ix[294]
print(cal(sample, 1))


def deal():
    for i in catalog[catalog.Z1 < 0.05].index:
        print(i)
        ct = catalog.ix[i]
        a, b, c, d, e, f = cal(ct, 1)
        catalog.at[i, 'init1'] = a
        catalog.at[i, 'con1'] = b
        catalog.at[i, 'adp1'] = c
        catalog.at[i, 'sig1'] = d
        catalog.at[i, 'geq1'] = e
        catalog.at[i, 'leq1'] = f
    for i in catalog[catalog.Z1 < 0.05].index:
        print(i)
        ct = catalog.ix[i]
        a, b, c, d, e, f = cal(ct, 2)
        catalog.at[i, 'init2'] = a
        catalog.at[i, 'con2'] = b
        catalog.at[i, 'adp2'] = c
        catalog.at[i, 'sig2'] = d
        catalog.at[i, 'geq2'] = e
        catalog.at[i, 'leq2'] = f

    catalog.to_csv('data.csv', sep=' ', columns=['NAME1', 'RA1', 'DEC1', 'Z1', 'init1', 'con1', 'adp1', 'sig1', 'geq1', 'leq1',
                                                 'NAME2', 'RA2', 'DEC2', 'Z2', 'init2', 'con2', 'adp2', 'sig2', 'geq2', 'leq2'])
