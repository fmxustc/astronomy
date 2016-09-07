import image_ops
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
        # eb = np.where(seg == 0, zero, arr)
        I = np.where(np.where((abs(seg-flag) > 1) | ((abs(seg180-flag) > 1) & (seg180 != 0) | (seg == 0)), zero, arr))
        I180 = np.rot90(I, 2)
        return np.sum(np.abs(I-I180))/np.sum(np.abs(I))
    # print(ctl.ix[[0,111]])95824.97+103
    # print(type(ctl.ix[flag].fi))
    # image_ops.show(np.where(float(ctl.ix[seg].fi) > 2, zero, adp))
    # eb = np.where(seg == 0, zero, adp)
    # og = np.where((abs(seg-flag) > 1) | ((abs(seg180-flag) > 1) & seg180), zero, eb)
    image_ops.show(np.where((abs(seg-flag) > 1) | ((abs(seg180-flag) > 1) & (seg180 != 0) | (seg == 0)), zero, adp))
    # image_ops.show(adp)
    return asym(adp)
    # return asym(init), asym(con), asym(adp), asym(sig), asym(geq), asym(leq)


warnings.filterwarnings('ignore')
path = '/home/franky/Desktop/type1/'
name = 'J083732.70+284218.7'
adpt = 'adaptive_equalization'
sigmoid = 'adjust_sigmoid'
gleq = 'global_equalization'
loeq = 'local_equalization'
sns.set_style('white')
catalog = pd.read_csv('data.csv', sep=' ')
# sample = catalog[catalog.Z1 < 0.05]
# sample = sample.ix[::5]
# for i in sample.index:
#     # print(i)
#     ct = catalog.ix[i]
#     a = cal(ct, 1)
#     sample.at[i, 'init'] = a
#
# sample.to_csv('qwe.csv', sep=',', columns=['NAME1', 'init'])
sd = pd.read_csv('qwe.csv')
# sd2 = pd.read_csv('qwer.csv')
# print(sd)
print(sd[sd.init > 0.5])
# sns.distplot(sd.init, kde=False, color='r', bins=np.linspace(0,1,51))
# sns.distplot(sd2.init, kde=False, bins=np.linspace(0,1,51))
# plt.show()
print(cal(catalog.ix[475], 1))


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

