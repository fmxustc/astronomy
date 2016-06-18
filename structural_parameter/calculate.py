import image_ops
import warnings


warnings.filterwarnings('ignore')
path = '/home/franky/Desktop/type1/'
init, seg = image_ops.load(path+'image/J033013.26-053236.1_r.fits', path+'seg/J033013.26-053236.1_r.fits', 52.555265,-5.5433500)
# resc = image_ops.contrast_stretching(init, 2, 98)
resc = image_ops.rescale(init, method='local_equalization', log_scale=True)
image_ops.show(image_ops.contrast_stretching(resc, 2, 98))

