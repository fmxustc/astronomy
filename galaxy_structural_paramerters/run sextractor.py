import subprocess
import platform
import pandas as pd
import warnings
import time
from termcolor import cprint as log


if __name__ == '__main__':

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
    type1_background_directory = settings.ix['type1_background', 'path']
    type2_background_directory = settings.ix['type2_background', 'path']
    type2_catalog_directory = settings.ix['type2_catalog', 'path']
    type2_seg_directory = settings.ix['type2_seg', 'path']
    type2_fits_directory = settings.ix['type2_fits', 'path']
    shell = settings.ix['shell', 'path']
    ds9 = settings.ix['ds9', 'path']
    sex = settings.ix['sex', 'path']

    warnings.filterwarnings('ignore')
    start_time = time.clock()

    catalog = pd.read_csv(settings.ix['catalog', 'path'])
    detected_set1 = []
    detected_set2 = []

    for k in catalog.index:
        t0 = time.clock()
        ctl = catalog.ix[k]
        conf1 = '-CATALOG_NAME %s -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s' % (type1_catalog_directory+ctl.NAME1+'_r.txt', type1_background_directory+ctl.NAME1+'_r.fits')
        conf2 = '-CATALOG_NAME %s -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s' % (type2_catalog_directory+ctl.NAME2+'_r.txt', type2_background_directory+ctl.NAME2+'_r.fits')
        if ctl.NAME1 not in detected_set1:
            _sp = subprocess.Popen('%s %s %s' % (sex, conf1, type1_fits_directory+ctl.NAME1+'_r.fits'), shell=True, executable=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            _sp.wait()
            detected_set1.append(ctl.NAME1)
        if ctl.NAME2 not in detected_set2:
            _sp = subprocess.Popen('%s %s %s' % (sex, conf2, type2_fits_directory+ctl.NAME2+'_r.fits'), shell=True, executable=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            _sp.wait()
            detected_set2.append(ctl.NAME2)
        t1 = time.clock()
        log('INDEX==> %d' % k, 'cyan', end='   ', attrs=['bold'])
        log('OBJECT==> %s %s' % (ctl.NAME1, ctl.NAME2),
            'green',
            end='    ')
        log('processed in %f seconds' % (t1 - t0), 'blue')

    end_time = time.clock()
    log('@The function takes %f seconds to complete' % (end_time - start_time),
        'grey',
        attrs=['bold'])
