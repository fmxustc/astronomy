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


def cal(agn_type, band, name, center, rms):

    def aysm(lmn):
        lnm = np.rot90(lmn, 2)
        return np.sum(np.abs(lmn - lnm)) / (2 * np.sum(np.abs(lmn)))

    def r(percent):
        box = min(radius, 1.5*petrosian_radius)
        ratio_msb = np.array(msb[:box])/msb[box-1]
        return int(np.argwhere(ratio_msb > percent/100)[0])

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
    # pollution = pd.read_csv(path['catalog'] + name + '_%s.txt' % band,
    #                         header=None,
    #                         sep='\s+',
    #                         names=['mag', 'x', 'y', 'fi', 'fp'])
    # pollution.index = range(1, len(pollution)+1)

    wx, wy = center[0], center[1]
    px, py = np.round(wcs.WCS(header).wcs_world2pix(wx, wy, 0))
    core = np.copy(luminous[py-10:py+11, px-10:px+11])
    py += core.argmax()//21-10
    px += core.argmax() % 21-10

    y, x = np.where(seg == seg[py][px])
    flux = luminous[y, x]
    n = len(flux)
    arg = np.argsort(flux)[::-1]
    cumulative_flux = np.copy(flux[arg])
    for k in range(1, n):
        cumulative_flux[k] += cumulative_flux[k-1]

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

    gini_flux = flux[arg][::-1]
    gini = np.sum((2 * np.arange(n) - n + 1) * np.abs(gini_flux)) / (np.mean(gini_flux) * n * (n - 1))

    try:
        cr = np.argwhere(surface_brightness < rms * 5)[0]
        concentration_f = float(msb[int(0.3 * cr)] / msb[cr])
    except IndexError:
        concentration_f = np.nan
    concentration_l = 5*np.log10(r(80)/r(20))

    moment_flux = np.array(dist[arg]**2 * flux[arg])
    moment = np.log10(np.sum(moment_flux[:np.argwhere(cumulative_flux > 0.2*cumulative_flux[-1])[0]]) / np.sum(moment_flux))

    # lm --->ib ln --->og
    lm = np.copy(luminous)
    ln = np.copy(luminous)
    for i in np.arange(py - petrosian_radius * 2,
                       py + petrosian_radius * 2 + 1):
        for j in np.arange(px - petrosian_radius * 2,
                           px + petrosian_radius * 2 + 1):
            if np.sqrt((i - py)**2 + (j - px)**2) <= petrosian_radius * 2:
                if seg[i][j] and seg[i][j] - seg[py][px]:
                        lm[i][j] = lm[2 * py - i][2 * px - j] = 0
                if seg[i][j] - seg[py][px]:
                    ln[i][j] = ln[2 * py - i][2 * px - j] = 0
            else:
                lm[i][j] = 0
                ln[i][j] = 0
    ib = np.copy(
          lm[py - 2 * petrosian_radius:py + 2 * petrosian_radius + 1,
             px - 2 * petrosian_radius:px + 2 * petrosian_radius + 1])
    og = np.copy(
          ln[py - 2 * petrosian_radius:py + 2 * petrosian_radius + 1,
             px - 2 * petrosian_radius:px + 2 * petrosian_radius + 1])
    os.system('rm %s' % (path['bound'] + name + '_%s_ib.fits' % band))
    ft.writeto(path['bound'] + name + '_%s_ib.fits' % band,
               lm)
    os.system('rm %s' % (path['bound'] + name + '_%s_og.fits' % band))
    ft.writeto(path['bound'] + name + '_%s_og.fits' % band,
               ln)

    return petrosian_radius, gini, moment, concentration_f, concentration_l, aysm(ib), aysm(og)


if __name__ == '__main__':

    def run():
        catalog = pd.read_csv('new_list.csv')
        # catalog = catalog
        catalog.index = range(len(catalog))

        calculated_set1 = {}
        calculated_set2 = {}

        for i in range(len(catalog)):
            log('INDEX==> %d' % i, 'cyan', end='   ', attrs=['bold'])
            t0 = time.clock()
            ctl = catalog.ix[i]
            log('OBJECT1==> %s' % ctl.NAME1, 'green', end='   ')
            if ctl.NAME1 not in calculated_set1:
                r, g, m, cf, cl, aib, aog = cal('type1', 'r', ctl.NAME1, [ctl.RA1, ctl.DEC1], ctl.RMS1)
                catalog.at[i, 'G1'] = g
                catalog.at[i, 'M1'] = m
                catalog.at[i, 'Cf1'] = cf
                catalog.at[i, 'Cl1'] = cl
                catalog.at[i, 'Aib1'] = aib
                catalog.at[i, 'Aog1'] = aog
                catalog.at[i, 'R1'] = r
                calculated_set1[ctl.NAME1] = i
            else:
                j = calculated_set1[ctl.NAME1]
                catalog.at[i, 'Aib1'] = catalog.at[j, 'Aib1']
                catalog.at[i, 'Aog1'] = catalog.at[j, 'Aog1']
                catalog.at[i, 'G1'] = catalog.at[j, 'G1']
                catalog.at[i, 'M1'] = catalog.at[j, 'M1']
                catalog.at[i, 'Cf1'] = catalog.at[j, 'Cf1']
                catalog.at[i, 'Cl1'] = catalog.at[j, 'Cl1']
                catalog.at[i, 'R1'] = catalog.at[j, 'R1']
            log('OBJECT2==> %s' % ctl.NAME2, 'green', end='    ')
            if ctl.NAME2 not in calculated_set2:
                r, g, m, cf, cl, aib, aog = cal('type2', 'r', ctl.NAME2, [ctl.RA2, ctl.DEC2], ctl.RMS2)
                catalog.at[i, 'G2'] = g
                catalog.at[i, 'M2'] = m
                catalog.at[i, 'Cf2'] = cf
                catalog.at[i, 'Cl2'] = cl
                catalog.at[i, 'Aib2'] = aib
                catalog.at[i, 'Aog2'] = aog
                catalog.at[i, 'R2'] = r
                calculated_set2[ctl.NAME2] = i
            else:
                j = calculated_set2[ctl.NAME2]
                catalog.at[i, 'Aib2'] = catalog.at[j, 'Aib2']
                catalog.at[i, 'Aog2'] = catalog.at[j, 'Aog2']
                catalog.at[i, 'G2'] = catalog.at[j, 'G2']
                catalog.at[i, 'M2'] = catalog.at[j, 'M2']
                catalog.at[i, 'Cf2'] = catalog.at[j, 'Cf2']
                catalog.at[i, 'Cl2'] = catalog.at[j, 'Cl2']
                catalog.at[i, 'R2'] = catalog.at[j, 'R2']
            t1 = time.clock()

            log('processed in %f seconds' % (t1 - t0), 'blue')

        catalog.to_csv('data.csv',
                       columns=['NAME1', 'RA1', 'DEC1', 'Z1', 'RMS1', 'R1', 'G1', 'M1', 'Cf1', 'Cl1', 'Aib1', 'Aog1',
                                'NAME2', 'RA2', 'DEC2', 'Z2', 'RMS2', 'R2', 'G2', 'M2', 'Cf2', 'Cl2', 'Aib2', 'Aog2'],
                       index=None,
                       sep=' ',
                       float_format='%.6f')
        return

    def test():
        print(cal('type1', 'r', 'J083732.70+284218.7', [129.386270, 28.705196], 0.015562))
        return

    def show():
        data = pd.read_csv('data.csv', sep=' ')
        print(len(data[(data.Z1 < 0.05) & (data.Aib1 > 0.5)]), len(data[(data.Z1 < 0.05) & (data.Aib2 > 0.5)]))
        bins = np.linspace(0.025, 1, 40)
        # sns.distplot(np.log10(data[data.Z1 < 0.05].M1), color='b', hist=True, kde=False, hist_kws={'color': 'b', 'histtype': "step", 'alpha': 1, "linewidth": 1.5})
        sns.distplot(data[(data.Z1 < 0.05)].Aib1, bins=bins, color='b', hist=True, kde=False, hist_kws={'color': 'b', 'histtype': "step", 'alpha': 1, "linewidth": 1.5})
        # sns.distplot(np.log10(data[data.Z1 < 0.05].M2), color='r', hist=True, kde=False, hist_kws={'color': 'r', 'histtype': "step", 'alpha': 1, "linewidth": 1.5})
        sns.distplot(data[(data.Z1 < 0.05)].Aib2, bins=bins, color='r', hist=True, kde=False, hist_kws={'color': 'r', 'histtype': "step", 'alpha': 1, "linewidth": 1.5})
        plt.show()
        return

    def pic():
        data = pd.read_csv('data.csv', sep=' ')
        sample = data[(data.Z1 < 0.05) & (data.M1 > 0)]
        # sample = data
        print(len(sample))
        sample.index = range(len(sample))
        objs = ''
        for t in range(1):
            for k in range(t*30, len(sample)):
                x = sample.ix[k]
                print(x.NAME2)
                # objs = objs + type1_bound_directory + x.NAME1 + '_r.fits '
                objs = objs + type2_bound_directory + x.NAME2 + '_r_og.fits '
            view = '-geometry 1920x1080 -view layout vertical -view panner no -view buttons no -view info yes -view magnifier no -view colorbar no'
            stts = '-invert -cmap value 1.75 0.275 -zoom 0.5 -minmax -log'
            subprocess.Popen('%s %s %s %s' % (ds9, view, stts, objs), shell=True, executable='/bin/zsh')
        return

    def diff():

        fig, ar = plt.subplots(3, 3, figsize=(16, 12), sharex=False, sharey=True)
        fig.suptitle('R Band Differences(z<0.05)')
        data = pd.read_csv('data.csv', sep=' ')

        b00 = np.linspace(0.3, 1, 21)
        sns.distplot(data[data.Z1 < 0.05].G1.values, bins=b00, ax=ar[0, 0], kde=False, hist=True, axlabel='Gini',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'b'})
        sns.distplot(data[data.Z1 < 0.05].G2.values, bins=b00, ax=ar[0, 0], kde=False, hist=True, axlabel='Gini',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'r'})
        ar[0, 0].legend(['type1', 'type2'])
        ar[0, 0].set_ylabel('Count')

        b01 = np.linspace(0.05, 0.85, 21)
        sns.distplot(data[data.Z1 < 0.05].Cf1.values, bins=b01, ax=ar[0, 1], kde=False, hist=True, axlabel='Concentration_f',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'b'})
        sns.distplot(data[data.Z1 < 0.05].Cf2.values, bins=b01, ax=ar[0, 1], kde=False, hist=True, axlabel='Concentration_f',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'r'})
        ar[0, 1].legend(['type1', 'type2'])

        b02 = np.linspace(1.5, 5.5, 26)
        sns.distplot(data[data.Z1 < 0.05].Cl1.values, bins=b02, ax=ar[0, 2], kde=False, hist=True, axlabel='Concentration_l',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'b'})
        sns.distplot(data[data.Z1 < 0.05].Cl2.values, bins=b02, ax=ar[0, 2], kde=False, hist=True, axlabel='Concentration_l',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'r'})
        ar[0, 2].legend(['type1', 'type2'])

        b10 = np.linspace(-3, -1, 21)
        sns.distplot(data[data.Z1 < 0.05].M1.values, bins=b10, ax=ar[1, 0], kde=False, hist=True, axlabel='Moment',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'b'})
        sns.distplot(data[data.Z1 < 0.05].M2.values, bins=b10, ax=ar[1, 0], kde=False, hist=True, axlabel='Moment',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'r'})
        ar[1, 0].legend(['type1', 'type2'])
        ar[1, 0].set_ylabel('Count')

        b20 = np.linspace(0, 0.6, 21)
        sns.distplot(data[data.Z1 < 0.05].Aib1.values, bins=b20, ax=ar[2, 0], kde=False, hist=True, axlabel='Asymmetry(including background pixels)',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'b'})
        sns.distplot(data[data.Z1 < 0.05].Aib2.values, bins=b20, ax=ar[2, 0], kde=False, hist=True, axlabel='Asymmetry(excluding background pixels)',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'r'})
        ar[2, 0].legend(['type1', 'type2'])
        ar[2, 0].set_ylabel('Count')

        # print(len(data[(data.Z1 < 0.05) ]))
        # print(len(data[(data.Z1 < 0.05) & (data.Aog1 > 0.15)]))
        # print(len(data[(data.Z1 < 0.05) & (data.Aog2 > 0.15)]))
        b21 = np.linspace(0, 0.3, 21)
        sns.distplot(data[data.Z1 < 0.05].Aog1.values, bins=b21, ax=ar[2, 1], kde=False, hist=True, axlabel='Asymmetry(including background pixels)',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'b'})
        sns.distplot(data[data.Z1 < 0.05].Aog2.values, bins=b21, ax=ar[2, 1], kde=False, hist=True, axlabel='Asymmetry(excluding background pixels)',
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": 'r'})
        ar[2, 1].legend(['type1', 'type2'])
        ar[2, 1].annotate('value>0.15\ntype1---3.06%\ntype2---8.06%', xy=(0.17, 70), bbox=dict(boxstyle="round", fc="1."),)

        plt.show()
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

    # run()
    # test()
    # show()
    # pic()
    diff()

    end_time = time.clock()
    log('@The function takes %f seconds to complete' % (end_time - start_time),
        'grey',
        attrs=['bold'])
