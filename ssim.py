import numpy as np
import tensorflow as tf
from PIL import Image
import math
from skimage.measure import compare_ssim


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def SSIM(y_pred, y_true):
    return compare_ssim(y_pred, y_true, multichannel=True)


def read_image(path):
    with tf.gfile.FastGFile(str(path), 'rb') as im:
        im = im.read()
        im = tf.image.decode_png(im, channels=3)
        im = tf.image.convert_image_dtype(im, dtype=tf.float32)
        return im


x_ = read_image('x.png')
y_ = read_image('y.png')
img_ = read_image('img.png')

psnr_x = tf.image.psnr(x_, y_, max_val=1)
ssim_x = tf.image.ssim(x_, y_, max_val=1)
psnr_img = tf.image.psnr(img_, y_, max_val=1)
ssim_img = tf.image.ssim(img_, y_, max_val=1)

x = np.array(Image.open('x.png'))
y = np.array(Image.open('y.png'))
img = np.array(Image.open('img.png'))

sess = tf.Session()

print('\nTensorflow -- \n')
print('PSNR = {}, SSIM = {}'.format(sess.run(psnr_x), sess.run(ssim_x)))
print('PSNR = {}, SSIM = {}'.format(sess.run(psnr_img), sess.run(ssim_img)))
print('\nMine -- \n')
print('PSNR = {}, SSIM = {}'.format(PSNR(x, y), SSIM(x, y)))
print('PSNR = {}, SSIM = {}'.format(PSNR(img, y), SSIM(img, y)))
