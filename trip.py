from __future__ import print_function
import os, sys
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import datetime

model_fn = 'tensorflow_inception_graph.pb'
output_size = 768
tile_size = 768
obj_switch_step = 75
save_step = 1
zoom_step = 1
zoom_px = 3
iter_n = 250
iter_rate = 0.06
timeline = []

# Create output folder
now = datetime.datetime.now()
output_dir = now.strftime('output/%Y%m%d_%H%M%S')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

# print('layers', layers)
# print('feature_nums', feature_nums)
# print('Number of layers', len(layers))
# print('Total number of feature channels:', sum(feature_nums))

# Fill timeline
for i in range(len(layers)):
    layer = layers[i].split('/')[1]
    num_features = feature_nums[i]

    if '3x3_bottleneck_pre_relu' not in layer:
        continue

    for j in range(num_features):
        timeline.append((layer, j))

# print([n.name for n in tf.get_default_graph().as_graph_def().node])
# asd = graph.get_tensor_by_name("import/output2:0")
# print(timeline)
# sys.exit(0)

# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def save_jpeg(jpeg_file, image):
    pil_image = PIL.Image.fromarray(image)
    pil_image.save(jpeg_file)

def normalize_image(image):
    image = np.clip(image, 0, 1)
    image = np.uint8(image * 255)
    return image

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=tile_size):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

def get_grad(i):
    current = timeline[i / obj_switch_step]
    t_score = tf.reduce_mean(T(current[0])[:,:,:,current[1]])
    t_grad = tf.gradients(t_score, t_input)[0]
    return t_grad

def render_lapnorm(img0=img_noise, visfunc=visstd, step=iter_rate, lap_n=4):
    iter_n = obj_switch_step * len(timeline)
    print('Estimated iteration: {}'.format(iter_n))

    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    img = img0.copy()
    for i in range(iter_n):
        # g = calc_grad_tiled(img, t_grad)
        g = calc_grad_tiled(img, get_grad(i))
        g = lap_norm_func(g)
        img += g*step
        print('Iteration: {}'.format(i + 1))

        if i > 0 and i % zoom_step == 0:
            new_size = output_size + (2 * zoom_px)
            img = resize(img, (new_size, new_size))
            img = img[zoom_px:zoom_px+output_size, zoom_px:zoom_px+output_size]

        if i % save_step == 0:
            print('Saving...')
            img_save = normalize_image(visstd(img))
            save_jpeg('{}/{}.jpg'.format(output_dir, str(i).zfill(10)), img_save)
            # save_jpeg('output.jpg'.format(output_dir, str(i).zfill(10)), img_save)

    # save_jpeg('final.jpg', normalize_image(visstd(img)))

img = np.random.uniform(size=(output_size, output_size, 3)) + 100.0
render_lapnorm(img)
