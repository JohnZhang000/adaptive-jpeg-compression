from cv2 import transpose
import numpy as np
import math
from scipy.fftpack import dct, idct, rfft, irfft
from skimage.transform import rescale, resize
import cv2
import random
import albumentations
import PIL
import PIL.Image
from io import BytesIO
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pickle
import joblib
import torch
from art.defences.preprocessor import GaussianAugmentation, JpegCompression,FeatureSqueezing,LabelSmoothing,Resample,SpatialSmoothing,ThermometerEncoding,TotalVarMin


import os
from my_spectrum_analyzer import img_spectrum_analyzer
import sys
sys.path.append('../common_code')
sys.path.append('./common_code')
import general as g

""""
PRO-TAT*: By adding randomness to the coefficients, integratS three basic affine transformations
, namely translation, scaling, and rotation, into one procedure.
*not published yet
T: translation limit
S: scaling limit
R: rotation limit
"""
def padding(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))
def cropping(img,cropx,cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]
def shifting(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def defend_PROTAT(img, T=0.16, S=0.16, R=4):
    # initialization
    angle = np.random.uniform(-R, R)
    scale = np.random.uniform(1 - S, 1 + S)
    dx = np.random.uniform(-T, T)
    dy = np.random.uniform(-T, T)
    height, width = img.shape[:2]
    center = (int(width / 2), int(height / 2))
    rotated_image = np.zeros_like(img)
    # translation
    shifted_image = shifting(img, int(dx * width), int(dy * height))
    # rotation
    for r in range(height):
        for c in range(width):
            #  apply rotation matrix
            y = (r - center[0]) * math.cos(angle * np.pi / 180.0) + (c - center[1]) * math.sin(angle * np.pi / 180.0)
            x = -(r - center[0]) * math.sin(angle * np.pi / 180.0) + (c - center[1]) * math.cos(angle * np.pi / 180.0)

            #  add offset
            y += center[0]
            x += center[1]

            #  get nearest index
            # a better way is linear interpolation
            x = round(x)
            y = round(y)

            # print(r, " ", c, " corresponds to-> " , y, " ", x)

            #  check if x/y corresponds to a valid pixel in input image
            if (x >= 0 and y >= 0 and x < width and y < height):
                rotated_image[r][c][:] = shifted_image[y][x][:]
    # scaling
    new_image = rescale(rotated_image, scale)
    if scale > 1:
        # center crop (original)
        scaled_image = cropping(new_image, img.shape[0], img.shape[1])
    else:
        # center padding (original)
        scaled_image = padding(new_image, img.shape[0], img.shape[1])
    return scaled_image


def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()

# GD algorithm that runs in the session (for EOT)
def tctensorGD(img,num_steps = 10,distort_limit = 0.25):
    xsteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    ysteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur
    xx = np.round(xx).astype(int)
    yy = np.round(yy).astype(int)
    xx[xx >= img.shape[0]] = img.shape[0]-1
    yy[yy >= img.shape[1]] = img.shape[1]-1
    map_x, map_y = np.meshgrid(xx, yy)
    # to speed up the mapping procedure, OpenCV 2 is adopted
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    # outimg = cv2.remap(img, map1=map_x, map2=map_y, interpolation=1, borderMode=4, borderValue=None)

    # xx = tf.cast(tf.clip_by_value(tf.round(listvec_x), 0, 298), tf.int32)
    # map_x = tf.tile(xx[:, 1:], (299, 1))
    # xx2 = tf.reverse((298 * tf.ones_like(xx, dtype=tf.int32) - xx), [1])
    # map_x2 = tf.tile(xx2[:, :299], (299, 1))
    # prev = tf.constant(0.0)
    # listvec_y = tf.zeros((1, 1))
    # for i in range(num_steps + 1):
    #     start = tf.cast(ys[i], tf.int32)
    #     end = tf.cast(ys[i], tf.int32) + y_step
    #     cur = tf.cond(end > width, lambda: tf.cast(height, tf.float32),
    #                   lambda: prev + tf.cast(y_step, tf.float32) * ystep[i])
    #     end = tf.cond(end > width, lambda: width, lambda: end)
    #     listvec_y = tf.concat([listvec_y, tf.reshape(tf.linspace(prev, cur, end - start), (1, -1))], -1)
    #     prev = cur
    # yy = tf.cast(tf.clip_by_value(tf.round(listvec_y), 0, 298), tf.int32)
    # map_y = tf.tile(tf.transpose(yy)[1:, :], (1, 299))
    # yy2 = tf.reverse((298 * tf.ones_like(yy, dtype=tf.int32) - yy), [1])
    # map_y2 = tf.tile(tf.transpose(yy2)[:299, :], (1, 299))
    # index_x = tf.cond(leftflag[0] > 0.5, lambda: tf.identity(map_x), lambda: tf.identity(map_x2))
    # index_y = tf.cond(upflag[0] > 0.5, lambda: tf.identity(map_y), lambda: tf.identity(map_y2))
    index = np.stack([map_y, map_x], 2)
    x_gd = gather_nd(img, torch.from_numpy(index))
    return x_gd

"""
RDG: Random distortion over grids.
Qiu, Han, et al. "Mitigating Advanced Adversarial Attacks with More Advanced Gradient Obfuscation Techniques." arXiv preprint arXiv:2005.13712 (2020).

num_steps: number of grids
distort_limit: distortion limit
"""
def defend_RDG(img,num_steps = 26,distort_limit = 0.33):
    xsteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    ysteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur
    xx = np.round(xx).astype(int)
    yy = np.round(yy).astype(int)
    xx[xx >= img.shape[0]] = img.shape[0]-1
    yy[yy >= img.shape[1]] = img.shape[1]-1
    map_x, map_y = np.meshgrid(xx, yy)
    # to speed up the mapping procedure, OpenCV 2 is adopted
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    outimg = cv2.remap(img, map1=map_x, map2=map_y, interpolation=1, borderMode=4, borderValue=None)
    return outimg

"""
CROP*: Random sized cropping

minlimit: the maximum scale that the img will remain
w2h: aspect ratio of crop.
*not published yet
"""
def get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2
def random_crop(img, crop_height, crop_width, h_start, w_start):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img

def defend_CROP(img,minlimit=0.66,w2h=0.91):
    crop_height = random.randint(int(img.shape[1]*minlimit),img.shape[1])
    crop_width = int(crop_height*w2h)
    h_start = random.random()
    w_start = random.random()
    return resize(random_crop(img, crop_height, crop_width, h_start, w_start),img.shape)


"""
RAND*: Random padding
*not published yet

scalimit: the maximum scale
"""
def defend_RAND(img,scalimit=1.3):
    maxvalue = np.int(img.shape[0] * scalimit)
    rnd = np.random.randint(img.shape[0],maxvalue,(1,))[0]
    rescaled = resize(img,(rnd,rnd))
    h_rem = maxvalue - rnd
    w_rem = maxvalue - rnd
    pad_left = np.random.randint(0,w_rem,(1,))[0]
    pad_right = w_rem - pad_left
    pad_top = np.random.randint(0,h_rem,(1,))[0]
    pad_bottom = h_rem - pad_top
    padded = np.pad(rescaled,((pad_top,pad_bottom),(pad_left,pad_right),(0,0)),'constant',constant_values = 0.5)
    padded = resize(padded,(img.shape[0],img.shape[0]))
    return padded

"""
FD: Feature Distillation
Liu, Zihao, et al. "Feature distillation: Dnn-oriented jpeg compression against adversarial examples." 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019.
"""
T = np.array([
        [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
        [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
        [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
        [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
        [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
        [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
        [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
        [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]
    ])
num = 8
q_table = np.ones((num,num))*30
q_table[0:4,0:4] = 25
def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
def rfft2 (block):
    return rfft(rfft(block.T).T)
def irfft2(block):
    return irfft(irfft(block.T).T)
def FD_fuction_sig(img):
    output = []
    input_matrix = img * 255

    h = input_matrix.shape[0]
    w = input_matrix.shape[1]
    c = input_matrix.shape[2]
    horizontal_blocks_num = w / num
    output2 = np.zeros((c, h, w))
    vertical_blocks_num = h / num

    c_block = np.split(input_matrix, c, axis=2)
    j = 0
    for ch_block in c_block:
        vertical_blocks = np.split(ch_block, vertical_blocks_num, axis=0)
        k = 0
        for block_ver in vertical_blocks:
            hor_blocks = np.split(block_ver, horizontal_blocks_num, axis=1)
            m = 0
            for block in hor_blocks:
                block = np.reshape(block, (num, num))
                block = dct2(block)
                # quantization
                table_quantized = np.matrix.round(np.divide(block, q_table))
                table_quantized = np.squeeze(np.asarray(table_quantized))
                # de-quantization
                table_unquantized = table_quantized * q_table
                IDCT_table = idct2(table_unquantized)
                if m == 0:
                    output = IDCT_table
                else:
                    output = np.concatenate((output, IDCT_table), axis=1)
                m = m + 1
            if k == 0:
                output1 = output
            else:
                output1 = np.concatenate((output1, output), axis=0)
            k = k + 1
        output2[j] = output1
        j = j + 1

    output2 = np.transpose(output2, (1, 0, 2))
    output2 = np.transpose(output2, (0, 2, 1))
    output2 = output2 / 255
    output2 = np.clip(np.float32(output2), 0.0, 1.0)
    return output2
def padresult_sig(cleandata):
    pad = albumentations.augmentations.transforms.PadIfNeeded(min_height=cleandata.shape[0]-cleandata.shape[0]%num+num, min_width=cleandata.shape[0]-cleandata.shape[0]%num+num, border_mode=4)
    paddata = pad(image=cleandata)['image']
    return paddata
def cropresult_sig(paddata,img):
    crop = albumentations.augmentations.crops.transforms.Crop(0, 0, img.shape[0], img.shape[1])
    resultdata = crop(image=paddata)['image']
    return resultdata

def defend_FD(img):
    paddata = padresult_sig(img)
    defendresult = FD_fuction_sig(paddata)
    resultdata = cropresult_sig(defendresult,img)
    return resultdata


"""
SHIELD: with randomized jpeg compression adopted for each blocks
Das, Nilaksh, et al. "Shield: Fast, practical defense and vaccination for deep learning using jpeg compression." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.
"""
def nearest_neighbour_scaling(label, new_h, new_w, patch_size, patch_n, patch_m):
    if len(label.shape) == 2:
        label_new = np.zeros([new_h, new_w])
    else:
        label_new = np.zeros([new_h, new_w, label.shape[2]])
    n_pos = np.arange(patch_n)
    m_pos = np.arange(patch_m)
    n_pos = n_pos.repeat(patch_size)[:299]
    m_pos = m_pos.repeat(patch_size)[:299]
    n_pos = n_pos.reshape(n_pos.shape[0], 1)
    n_pos = np.tile(n_pos, (1, new_w))
    m_pos = np.tile(m_pos, (new_h, 1))
    assert n_pos.shape == m_pos.shape
    label_new[:, :] = label[n_pos[:, :], m_pos[:, :]]
    return label_new
def jpeg(input_array, quali):
    pil_image = PIL.Image.fromarray((input_array * 255.0).astype(np.uint8))
    f = BytesIO()
    pil_image.save(f, format='jpeg', quality=quali)  # quality level specified in paper
    jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0
    return jpeg_image
def defend_SHIELD(x, qualities=(20, 40, 60, 80), patch_size=8):
    n = x.shape[0]
    m = x.shape[1]
    patch_n = int(n / patch_size)
    patch_m = int(m / patch_size)
    num_qualities = len(qualities)
    n = x.shape[0]
    m = x.shape[1]
    if n % patch_size > 0:
        patch_n = np.int(n / patch_size) + 1
        delete_n = 1
    if m % patch_size > 0:
        patch_m = np.int(m / patch_size) + 1
        delet_m = 1

    R = np.tile(np.reshape(np.arange(n), (n, 1)), [1, m])
    C = np.reshape(np.tile(np.arange(m), [n]), (n, m))
    # mini_Z = (np.random.rand(patch_n, patch_m) * num_qualities).astype(int)
    mini_Z = np.random.rand(patch_n, patch_m) * num_qualities
    mini_Z = mini_Z.astype(np.int32)
    Z = (nearest_neighbour_scaling(mini_Z, n, m, patch_size, patch_n, patch_m)).astype(int)
    indices = np.transpose(np.stack((Z, R, C)), (1, 2, 0))
    # x = img_as_ubyte(x)
    x_compressed_stack = []

    for quali in qualities:
        processed = jpeg(x, quali)
        x_compressed_stack.append(processed)

    x_compressed_stack = np.asarray(x_compressed_stack)
    x_slq = np.zeros((n, m, 3))
    for i in range(n):
        for j in range(m):
            x_slq[i, j] = x_compressed_stack[tuple(indices[i][j])]
    return x_slq


"""
BdR: Bit-depth Reduction
Xu, Weilin, David Evans, and Yanjun Qi. "Feature squeezing: Detecting adversarial examples in deep neural networks." arXiv preprint arXiv:1704.01155 (2017).
"""
def defend_BdR(arr, depth=3):
    arr = (arr * 255.0).astype(np.uint8)
    shift = 8 - depth
    arr = (arr >> shift) << shift
    arr = arr.astype(np.float32)/255.0
    return arr


"""
ET: Elastic Transformation
Simard, Patrice Y., David Steinkraus, and John C. Platt. "Best practices for convolutional neural networks applied to visual document analysis." Icdar. Vol. 3. No. 2003. 2003.
alpha:60
sigma:10
aaf: 20
"""
def defend_ET(img):
    aug = albumentations.ElasticTransform(p=1, alpha=60, sigma=10, alpha_affine=20)
    augmented = aug(image=img.astype(np.float32))
    auged = augmented['image']
    return auged


"""
MB*: Motion Blur, apply motion blur to the input image using a random-sized kernel (simulates motion blur)
*not published yet
"""
def defend_MB(img):
    aug = albumentations.MotionBlur(blur_limit=(3,9),p=1)
    augmented = aug(image=img.astype(np.float32))
    auged = augmented['image']
    return auged


"""
GB: Glass Blur, apply Glass blur/noise to the input image
Hendrycks, Dan, and Thomas Dietterich. "Benchmarking neural network robustness to common corruptions and perturbations." arXiv preprint arXiv:1903.12261 (2019).
"""
def defend_GB(img, sigma=0.9, max_delta = 1, iterations = 1):
    aug = albumentations.GlassBlur(sigma=sigma,max_delta=max_delta,iterations=iterations,p=1)
    augmented = aug(image=(img*255).astype(np.uint8))
    auged = augmented['image']/255
    return auged


"""
JPEG*: Randomized JPEG
*not published
"""
def defend_JPEG(img):
    aug = albumentations.JpegCompression(quality_lower=20, quality_upper=80,p=1)
    augmented = aug(image=(img*255).astype(np.uint8))
    auged = augmented['image']/255
    return auged


"""
WebP: Randomized WebP compression
http://developers.google.com/speed/webp/docs/compression
"""
def defend_WebP(img):
    aug = albumentations.ImageCompression(quality_lower=20,quality_upper=80,compression_type=0,p=1)
    augmented = aug(image=(img*255).astype(np.uint8))
    auged = augmented['image']/255
    return auged


"""
CD: Coarse Dropout, box dropout, dropout boxes cut from the original image
DeVries, Terrance, and Graham W. Taylor. "Improved regularization of convolutional neural networks with cutout." arXiv preprint arXiv:1708.04552 (2017).
"""
def defend_CD(img):
    aug = albumentations.CoarseDropout(p = 1)
    augmented = aug(image=img.astype(np.float32))
    auged = augmented['image']
    return auged


"""
GN*: randomly adding gaussian noise
*not published yet
"""
def defend_GN(img):
    aug =albumentations.GaussNoise(var_limit=(0.0005, 0.005), mean=0,p=1)
    augmented = aug(image=img.astype(np.float32))
    auged = augmented['image']
    return auged


"""
PD: pixel deflection
Prakash, Aaditya, et al. "Deflecting adversarial attacks with pixel deflection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
"""
def defend_PD(img, deflections=600, window=10):
    img = np.copy(img)
    H, W, C = img.shape
    while deflections > 0:
        #for consistency, when we deflect the given pixel from all the three channels.
        for c in range(C):
            x,y = random.randint(0,H-1), random.randint(0,W-1)
            while True: #this is to ensure that PD pixel lies inside the image
                a,b = random.randint(-1*window,window), random.randint(-1*window,window)
                if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
            # calling pixel deflection as pixel swap would be a misnomer,
            # as we can see below, it is one way copy
            img[x,y,c] = img[x+a,y+b,c]
        deflections -= 1
    return img

def defend_webpf(img):
    quality=80
    assert(img.shape[-3]==img.shape[-2])  
    aug = albumentations.Compose([
        albumentations.ImageCompression(quality_lower=quality,quality_upper=quality,compression_type=0,p=1),
        albumentations.HorizontalFlip(p=1.0)])
    
    augmented = aug(image=(img*255).astype(np.uint8))
    auged = augmented['image']/255
    return auged

def defend_webpf_wrap(img,labels=None):
    if isinstance(img,torch.Tensor): img=img.numpy()
    if img.ndim==3 and img.shape[-1]==img.shape[-2]: img=np.expand_dims(img.transpose(1,2,0),axis=0)
    elif img.ndim==4 and img.shape[-1]==img.shape[-2]: img=img.transpose(0,2,3,1)

    assert(img.shape[-3]==img.shape[-2])
    
    if img.ndim==3:
        auged = defend_webpf(img)
    elif img.ndim==4:
        auged_list=[]
        for i in range(img.shape[0]):
            auged_tmp = defend_webpf(img[i,...])
            auged_list.append(np.expand_dims(auged_tmp,axis=0))
        auged=np.vstack(auged_list)
    auged=auged.astype(np.float32)
    return auged,labels

def defend_webpf_my(img,eps):
    quality=82.49907*np.exp(-eps/0.57195)+19.01903
    # print(quality)
    quality=np.clip(np.round(quality),1,100)
    assert(img.shape[-3]==img.shape[-2])  
    aug = albumentations.Compose([
        albumentations.ImageCompression(quality_lower=quality,quality_upper=quality,compression_type=0,p=1),
        albumentations.HorizontalFlip(p=1.0)])
    
    augmented = aug(image=(img*255).astype(np.uint8))
    auged = augmented['image']/255
    return auged

def defend_webpf_my_wrap(img,eps,labels=None):
    assert(img.shape[-3]==img.shape[-2])
    
    if img.ndim==3:
        auged = defend_webpf(img)
    elif img.ndim==4:
        auged_list=[]
        for i in range(img.shape[0]):
            auged_tmp = defend_webpf_my(img[i,...],eps[i])
            auged_list.append(np.expand_dims(auged_tmp,axis=0))
        auged=np.vstack(auged_list)
    auged=auged.astype(np.float32)
    return auged,labels

class defend_my_webpf:

    # 解释器初始化
    def __init__(self,model_pkl,img_size,spectrum_size):
        self.reg=joblib.load(model_pkl)
        self.s_analyzer=img_spectrum_analyzer(img_size,spectrum_size).batch_get_spectrum_feature
        # if 'adaboost' in model_pkl:
        #     self.reg_type='adb'
        # elif 'svm' in model_pkl:
        #     self.reg_type='svm'
        # else:
        #     raise Exception('Wrong model type')
        
    def defend(self, img, labels=None):
        spectrum  = self.s_analyzer(img.transpose(0,3,1,2))
        pred_eps  = self.reg.predict(spectrum)
        auged,labels = defend_webpf_my_wrap(img,pred_eps)
        return auged,labels       
       
def defend_gaua_wrap(img,labels=None):
    if isinstance(img,torch.Tensor): img=img.numpy()
    if img.ndim==3 and img.shape[-1]==img.shape[-2]: img=np.expand_dims(img.transpose(1,2,0),axis=0)
    elif img.ndim==4 and img.shape[-1]==img.shape[-2]: img=img.transpose(0,2,3,1)
    assert(img.shape[-3]==img.shape[-2])
    
    GauA=GaussianAugmentation(sigma=0.01,augmentation=False)
    if img.ndim==3:
        auged = GauA(img)
    elif img.ndim==4:
        auged_list=[]
        for i in range(img.shape[0]):
            auged_tmp = GauA(img[i,...])
            auged_list.append(np.expand_dims(auged_tmp,axis=0))
        auged=np.vstack(auged_list)
    auged=auged.astype(np.float32)
    return auged,labels

def defend_bdr_wrap(img,labels=None):
    if isinstance(img,torch.Tensor): img=img.numpy()
    if img.ndim==3 and img.shape[-1]==img.shape[-2]: img=np.expand_dims(img.transpose(1,2,0),axis=0)
    elif img.ndim==4 and img.shape[-1]==img.shape[-2]: img=img.transpose(0,2,3,1)
    assert(img.shape[-3]==img.shape[-2])
    
    # GauA=GaussianAugmentation(sigma=0.01,augmentation=False)
    bdr=SpatialSmoothing()
    if img.ndim==3:
        auged = bdr(img)
    elif img.ndim==4:
        auged_list=[]
        for i in range(img.shape[0]):
            auged_tmp = bdr(img[i,...])
            auged_list.append(np.expand_dims(auged_tmp,axis=0))
        auged=np.vstack(auged_list)
    auged=auged.astype(np.float32)
    return auged,labels

def defend_rdg_wrap(img,labels=None):
    if isinstance(img,torch.Tensor): img=img.numpy()
    if img.ndim==3 and img.shape[-1]==img.shape[-2]: img=np.expand_dims(img.transpose(1,2,0),axis=0)
    elif img.ndim==4 and img.shape[-1]==img.shape[-2]: img=img.transpose(0,2,3,1)
    assert(img.shape[-3]==img.shape[-2])
    num_steps=4
    
    if img.ndim==3:
        auged = defend_RDG(img,num_steps=num_steps)
    elif img.ndim==4:
        auged_list=[]
        for i in range(img.shape[0]):
            auged_tmp = defend_RDG(img[i,...],num_steps=num_steps)
            auged_list.append(np.expand_dims(auged_tmp,axis=0))
        auged=np.vstack(auged_list)
    auged=auged.astype(np.float32)
    return auged,labels

def defend_fd_wrap(img,labels=None):
    if isinstance(img,torch.Tensor): img=img.numpy()
    if img.ndim==3 and img.shape[-1]==img.shape[-2]: img=np.expand_dims(img.transpose(1,2,0),axis=0)
    elif img.ndim==4 and img.shape[-1]==img.shape[-2]: img=img.transpose(0,2,3,1)
    assert(img.shape[-3]==img.shape[-2])
    
    if img.ndim==3:
        auged = defend_FD(img)
    elif img.ndim==4:
        auged_list=[]
        for i in range(img.shape[0]):
            auged_tmp = defend_FD(img[i,...])
            auged_list.append(np.expand_dims(auged_tmp,axis=0))
        auged=np.vstack(auged_list)
    auged=auged.astype(np.float32)
    return auged,labels

def defend_bdr_wrap(img,labels=None):
    if isinstance(img,torch.Tensor): img=img.numpy()
    if img.ndim==3 and img.shape[-1]==img.shape[-2]: img=np.expand_dims(img.transpose(1,2,0),axis=0)
    elif img.ndim==4 and img.shape[-1]==img.shape[-2]: img=img.transpose(0,2,3,1)
    assert(img.shape[-3]==img.shape[-2])
    
    if img.ndim==3:
        auged = defend_BdR(img)
    elif img.ndim==4:
        auged_list=[]
        for i in range(img.shape[0]):
            auged_tmp = defend_BdR(img[i,...])
            auged_list.append(np.expand_dims(auged_tmp,axis=0))
        auged=np.vstack(auged_list)
    auged=auged.astype(np.float32)
    return auged,labels

def defend_shield_wrap(img,labels=None):
    if isinstance(img,torch.Tensor): img=img.numpy()
    if img.ndim==3 and img.shape[-1]==img.shape[-2]: img=np.expand_dims(img.transpose(1,2,0),axis=0)
    elif img.ndim==4 and img.shape[-1]==img.shape[-2]: img=img.transpose(0,2,3,1)
    assert(img.shape[-3]==img.shape[-2])
    
    if img.ndim==3:
        auged = defend_SHIELD(img)
    elif img.ndim==4:
        auged_list=[]
        for i in range(img.shape[0]):
            auged_tmp = defend_SHIELD(img[i,...])
            auged_list.append(np.expand_dims(auged_tmp,axis=0))
        auged=np.vstack(auged_list)
    auged=auged.astype(np.float32)
    return auged,labels

def defend_jpeg_wrap(img,labels=None):
    if isinstance(img,torch.Tensor): img=img.numpy()
    if img.ndim==3 and img.shape[-1]==img.shape[-2]: img=np.expand_dims(img.transpose(1,2,0),axis=0)
    elif img.ndim==4 and img.shape[-1]==img.shape[-2]: img=img.transpose(0,2,3,1)
    assert(img.shape[-3]==img.shape[-2])

    auged_list=[]
    for i in range(img.shape[0]):
        pil_image = PIL.Image.fromarray((img[i]*255.0).astype(np.uint8))
        f = BytesIO()
        pil_image.save(f, format='jpeg', quality=75) # quality level specified in paper
        jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32)/255.0
        auged_list.append(jpeg_image)
    auged=np.vstack(auged_list)
    auged=auged.astype(np.float32)
    return auged,labels

def tctensorGD_warp(img,labels=None):
    assert isinstance(img,torch.Tensor)
    if len(img.shape)==3 and img.shape[-1]==img.shape[-2]: img=img.permute(1,2,0).unsqueeze(0)
    elif len(img.shape)==4 and img.shape[-1]==img.shape[-2]: img=img.permute(0,2,3,1)
    assert(img.shape[-3]==img.shape[-2])
    auged_list=[]
    for i in range(img.shape[0]):
        auged=tctensorGD(img[i])
        auged_list.append(auged.unsqueeze(0).permute(0,3,1,2))
    return torch.vstack(auged_list)
