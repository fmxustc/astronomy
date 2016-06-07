import platform
import pandas as pd
import warnings
import time
import astropy.io.fits as ft
import astropy.wcs as wcs
import numpy as np
from termcolor import cprint as log
import seaborn as sns
import os
import matplotlib.pyplot as plt


def cal(agn_type, band, name, center):

    path = {
        'image': type1_fits_directory if agn_type == 'type1' else type2_fits_directory,
        'seg': type1_seg_directory if agn_type == 'type1' else type2_seg_directory,
        'catalog': type1_catalog_directory if agn_type == 'type1' else type2_catalog_directory
    }

    hdu = ft.open(path['image']+name+'_%s.fits' % band)[0]
    header = hdu.header
    luminous = hdu.data
    segmentation = ft.open(path['seg']+name+'_%s.fits' % band)[0].data
    catalog = pd.read_csv(path['catalog']+name+'_%s.txt' % band, header=None, sep='\s+', names=['mag', 'x', 'y', 'fi', 'fp'])

    wx, wy = center[0], center[1]
    px, py = np.round(wcs.WCS(header).wcs_world2pix(wx, wy, 0))

    y, x = np.where(segmentation == segmentation[py][px])
    sy, sx = 2*py-y, 2*px-x
    flux = luminous[y, x]
    arg = np.argsort(flux)[::-1]
    seg = segmentation[y, x]

    dist = np.sqrt((y-py)**2+(x-px)**2)
    radius = np.ceil(dist.max())+1
    sb = np.zeros(radius)
    cnt = np.zeros(radius)
    for k in np.arange(len(dist)):
        sb[np.ceil(dist[k])] += flux[k]
        cnt[np.ceil(dist[k])] += 1
    # log(cnt[:20], 'blue')
    surface_brightness = sb/cnt
    msb = np.copy(sb)
    for k in np.arange(1, radius):
        msb[k] += msb[k-1]
        cnt[k] += cnt[k-1]
    mean_surface_brightness = msb/cnt
    # log(cnt[:20], 'red')
    eta = surface_brightness/mean_surface_brightness
    # sns.tsplot(eta[:300])
    # plt.show()

    petrosian_radius = float(np.argwhere(eta < 0.2)[0])
    for i in np.arange(py-petrosian_radius, py+petrosian_radius+1):
        for j in np.arange(px-petrosian_radius, px+petrosian_radius+1):
            if abs((i-py)**2+(j-px)**2-petrosian_radius**2) < 1e-03:
                luminous[i][j] = luminous[py][px]
    print(luminous[py-100:py+101, px-100:px+101])
    os.system('rm ./tmp.fits')
    ft.writeto('./tmp.fits', luminous[py-100:py+101, px-100:px+101])
    return
    # return 1, 1, 1, 1, 1

if __name__ == '__main__':

    def run():
        catalog = pd.read_csv('list.csv')
        catalog = catalog[catalog.Z1 <= 0.05]
        catalog.index = range(len(catalog))

        calculated_set1 = {}
        calculated_set2 = {}

        for i in range(len(catalog)):
            t0 = time.clock()
            ctl = catalog.ix[i]
            if ctl.NAME1 not in calculated_set1:
                r, g, m, a, c = cal(type1_fits_directory, ctl.NAME1, [ctl.RA1, ctl.DEC1])
                catalog.at[i, 'G1'] = g
                catalog.at[i, 'M1'] = m
                catalog.at[i, 'A1'] = a
                catalog.at[i, 'C1'] = c
                catalog.at[i, 'R1'] = r
                calculated_set1[ctl.NAME1] = i
            else:
                j = calculated_set1[ctl.NAME1]
                catalog.at[i, 'G1'] = catalog.at[j, 'G1']
                catalog.at[i, 'M1'] = catalog.at[j, 'M1']
                catalog.at[i, 'A1'] = catalog.at[j, 'A1']
                catalog.at[i, 'C1'] = catalog.at[j, 'C1']
                catalog.at[i, 'R1'] = catalog.at[j, 'R1']
            if ctl.NAME2 not in calculated_set2:
                r, g, m, a, c = cal(type2_fits_directory, ctl.NAME2,
                                    [ctl.RA2, ctl.DEC2])
                catalog.at[i, 'G2'] = g
                catalog.at[i, 'M2'] = m
                catalog.at[i, 'A2'] = a
                catalog.at[i, 'C2'] = c
                catalog.at[i, 'R2'] = r
                calculated_set2[ctl.NAME2] = i
            else:
                j = calculated_set2[ctl.NAME2]
                catalog.at[i, 'G2'] = catalog.at[j, 'G2']
                catalog.at[i, 'M2'] = catalog.at[j, 'M2']
                catalog.at[i, 'A2'] = catalog.at[j, 'A2']
                catalog.at[i, 'C2'] = catalog.at[j, 'C2']
                catalog.at[i, 'R2'] = catalog.at[j, 'R2']
            t1 = time.clock()
            log('INDEX==> %d' % i, 'cyan', end='   ', attrs=['bold'])
            log('OBJECT==> %s %s' % (ctl.NAME1, ctl.NAME2),
                'green',
                end='    ')
            log('processed in %f seconds' % (t1 - t0), 'blue')

        catalog.to_csv('data1.csv',
                       columns=['NAME1', 'R1', 'G1', 'M1', 'A1', 'C1', 'NAME2',
                                'R2', 'G2', 'M2', 'A2', 'C2'],
                       index_label=['INDEX'],
                       sep=' ',
                       float_format='%e')
        return

    def test():
        cal('type1', 'r', 'J083732.70+284218.7', [129.38627, 28.705196])
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
    type2_catalog_directory = settings.ix['type2_catalog', 'path']
    type2_seg_directory = settings.ix['type2_seg', 'path']
    type2_fits_directory = settings.ix['type2_fits', 'path']
    shell = settings.ix['shell', 'path']
    ds9 = settings.ix['ds9', 'path']
    sex = settings.ix['sex', 'path']

    warnings.filterwarnings('ignore')
    start_time = time.clock()

    # run()
    test()

    end_time = time.clock()
    log('@The function takes %f seconds to complete' % (end_time - start_time),
        'grey',
        attrs=['bold'])
