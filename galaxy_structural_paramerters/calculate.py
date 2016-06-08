import platform
import pandas as pd
import warnings
import time
import astropy.io.fits as ft
import astropy.wcs as wcs
import numpy as np
from termcolor import cprint as log
import matplotlib.pyplot as plt
import os
import subprocess
import seaborn as sns


def cal(agn_type, band, name, center):

    path = {
        'image': type1_fits_directory
        if agn_type == 'type1' else type2_fits_directory,
        'seg': type1_seg_directory
        if agn_type == 'type1' else type2_seg_directory,
        'bg': type1_background_directory
        if agn_type == 'type1' else type2_background_directory,
        'catalog': type1_catalog_directory
        if agn_type == 'type1' else type2_catalog_directory,
        'bound': type1_bound_directory
        if agn_type == 'type1' else type2_bound_directory
    }

    hdu = ft.open(path['image'] + name + '_%s.fits' % band)[0]
    header = hdu.header
    luminous = hdu.data
    seg = ft.open(path['seg'] + name + '_%s.fits' % band)[0].data
    catalog = pd.read_csv(path['catalog'] + name + '_%s.txt' % band,
                          header=None,
                          sep='\s+',
                          names=['mag', 'x', 'y', 'fi', 'fp'])
    catalog.index = range(1, len(catalog)+1)

    wx, wy = center[0], center[1]
    px, py = np.round(wcs.WCS(header).wcs_world2pix(wx, wy, 0))
    core = np.copy(luminous[py-10:py+11, px-10:px+11])
    py += core.argmax()//21-10
    px += core.argmax() % 21-10

    y, x = np.where(seg == seg[py][px])
    flux = luminous[y, x]
    # arg = np.argsort(flux)[::-1]

    dist = np.sqrt((y - py)**2 + (x - px)**2)
    radius = np.ceil(dist.max()) + 1
    sb = np.zeros(radius)
    cnt = np.zeros(radius)
    for k in np.arange(len(dist)):
        sb[np.ceil(dist[k])] += flux[k]
        cnt[np.ceil(dist[k])] += 1
    surface_brightness = sb / cnt
    msb = np.copy(sb)
    for k in np.arange(1, radius):
        msb[k] += msb[k - 1]
        cnt[k] += cnt[k - 1]
    mean_surface_brightness = msb / cnt
    eta = surface_brightness / mean_surface_brightness
    eta = np.copy(eta[~np.isnan(eta)])
    try:
        petrosian_radius = float(np.argwhere(eta < 0.2)[0])
    except IndexError:
        petrosian_radius = float(np.argmin(eta))

    for i in np.arange(py - petrosian_radius * 2,
                       py + petrosian_radius * 2 + 1):
        for j in np.arange(px - petrosian_radius * 2,
                           px + petrosian_radius * 2 + 1):
            if np.sqrt((i - py)**2 + (j - px)**2) <= petrosian_radius * 2:
                if seg[i][j] and seg[i][j] - seg[py][px]:
                        luminous[i][j] = luminous[2 * py - i][2 * px - j] = 0
            else:
                luminous[i][j] = 0
    _I = np.copy(
        luminous[py - 2 * petrosian_radius:py + 2 * petrosian_radius + 1,
                 px - 2 * petrosian_radius:px + 2 * petrosian_radius + 1])
    _I180 = np.rot90(_I, 2)
    os.system('rm %s' % (path['bound'] + name + '_%s.fits' % band))
    ft.writeto(path['bound'] + name + '_%s.fits' % band,
               luminous)

    return petrosian_radius, np.sum(np.abs(_I - _I180)) / np.sum(np.abs(_I))


if __name__ == '__main__':

    def run():
        catalog = pd.read_csv('list.csv')
        catalog = catalog
        catalog.index = range(len(catalog))

        calculated_set1 = {}
        calculated_set2 = {}

        for i in range(len(catalog)):
            log('INDEX==> %d' % i, 'cyan', end='   ', attrs=['bold'])
            t0 = time.clock()
            ctl = catalog.ix[i]
            log('OBJECT1==> %s' % ctl.NAME1, 'green', end='   ')
            if ctl.NAME1 not in calculated_set1:
                r, a = cal('type1', 'r', ctl.NAME1, [ctl.RA1, ctl.DEC1])
                catalog.at[i, 'A1'] = a
                catalog.at[i, 'R1'] = r
                calculated_set1[ctl.NAME1] = i
            else:
                j = calculated_set1[ctl.NAME1]
                catalog.at[i, 'A1'] = catalog.at[j, 'A1']
                catalog.at[i, 'R1'] = catalog.at[j, 'R1']
            log('OBJECT2==> %s' % ctl.NAME2, 'green', end='    ')
            if ctl.NAME2 not in calculated_set2:
                r, a = cal('type2', 'r', ctl.NAME2, [ctl.RA2, ctl.DEC2])
                catalog.at[i, 'A2'] = a
                catalog.at[i, 'R2'] = r
                calculated_set2[ctl.NAME2] = i
            else:
                j = calculated_set2[ctl.NAME2]
                catalog.at[i, 'A2'] = catalog.at[j, 'A2']
                catalog.at[i, 'R2'] = catalog.at[j, 'R2']
            t1 = time.clock()

            log('processed in %f seconds' % (t1 - t0), 'blue')

        catalog.to_csv('data_ib.csv',
                       columns=['NAME1', 'R1', 'A1', 'NAME2', 'R2', 'A2'],
                       index=None,
                       sep=' ',
                       float_format='%.6f')
        return

    def test():
        print('%6f %.6f' % cal('type2', 'r', 'J170622.22+212422.2',
                               [256.59260, 21.406173]))
        return

    def show():
        data = pd.read_csv('data.csv', sep=' ')
        print(data.A1.values.shape, data.A2.values.shape)
        bins = np.linspace(0, 1, 10)
        sns.distplot(data.A1, bins=bins, color='b', hist=True, kde=False, hist_kws={'color': 'b', 'histtype': "step", 'alpha': 1, "linewidth": 1.5})
        sns.distplot(data.A2, bins=bins, color='r', hist=True, kde=False, hist_kws={'color': 'r', 'histtype': "step", 'alpha': 1, "linewidth": 1.5})
        plt.show()
        return

    def pic():
        data = pd.read_csv('data.csv', sep=' ')
        sample = data[(data.A2 > 0.4) & (data.A2 < 0.5)]
        sample.index = range(len(sample))
        objs = ''
        for k in range(min(20, len(sample))):
            x = sample.ix[k]
            print(x.NAME2)
            objs = objs + type2_bound_directory + x.NAME2 + '_r.fits '
        view = '-geometry 1920x1080 -view layout vertical -view panner no -view buttons no -view info yes -view magnifier no -view colorbar no'
        stts = '-invert -cmap value 1.75 0.275 -zoom 0.5 -minmax -log'
        subprocess.Popen('%s %s %s %s' % (ds9, view, stts, objs), shell=True, executable='/bin/zsh')
        return

    if platform.system() == 'Linux':
        settings = pd.read_table('linux_setting.param',
                                 sep=' ',
                                 header=0,
                                 index_col='obj')
    else:
        settings = pd.read_table('mac_setting.param',
                                 sep=' ',
                                 header=0,
                                 index_col='obj')

    type1_fits_directory = settings.ix['type1_fits', 'path']
    type1_seg_directory = settings.ix['type1_seg', 'path']
    type1_catalog_directory = settings.ix['type1_catalog', 'path']
    type1_bound_directory = settings.ix['type1_bound', 'path']
    type1_background_directory = settings.ix['type1_background', 'path']
    type2_background_directory = settings.ix['type2_background', 'path']
    type2_bound_directory = settings.ix['type2_bound', 'path']
    type2_catalog_directory = settings.ix['type2_catalog', 'path']
    type2_seg_directory = settings.ix['type2_seg', 'path']
    type2_fits_directory = settings.ix['type2_fits', 'path']
    shell = settings.ix['shell', 'path']
    ds9 = settings.ix['ds9', 'path']
    sex = settings.ix['sex', 'path']

    warnings.filterwarnings('ignore')
    start_time = time.clock()

    run()
    # test()
    # show()
    # pic()

    end_time = time.clock()
    log('@The function takes %f seconds to complete' % (end_time - start_time),
        'grey',
        attrs=['bold'])
