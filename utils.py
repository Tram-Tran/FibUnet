import numpy as np
from matplotlib import pyplot as plt
import os
import math

import io
import imageio
from IPython.display import display
from ipywidgets import widgets, Layout, HBox

import skvideo.io
from PIL import Image
import cv2 as cv
import hickle as hkl

import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from tensorflow.image import ssim
from tensorflow.image import psnr

import gc

def mean_ssim(y_train, y_pred, max_val=1, filter_size=16):
    s = 0
    for i in range(y_train.shape[0]):
        s += tf.math.reduce_mean(
            ssim(y_train[i].astype('float32'), y_pred[i].astype('float32'), max_val=max_val, filter_size=filter_size)
            ).numpy()
        
    return s / float(y_train.shape[0])

def my_psnr(y_train, y_pred, max_val=1):
    mse = np.mean((y_train - y_pred) ** 2)
    if mse == 0: mse = 1e-8 # prevent undefined log value
    p = ((20 * math.log10(max_val)) - (10 * math.log10(mse)))
    return p

def mean_psnr(y_train, y_pred, max_val=1):
    p = 0
    for i in range(y_train.shape[0]):
        p += tf.math.reduce_mean(
            psnr(y_train[i].astype('float32'), y_pred[i].astype('float32'), max_val=max_val)
            ).numpy()
        # for j in range(y_train.shape[1]):
        #     p += my_psnr(y_train[i][j].astype('float32'), y_pred[i][j].astype('float32'), max_val=max_val)
        
    return p / float(y_train.shape[0])

def get_flops(model):
    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
    graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
    flops = graph_info.total_float_ops
    return flops

def load_vids(pathname, ext='.avi'):
    data = []

    for filename in os.listdir(pathname):
        if filename.endswith(ext):
            file = skvideo.io.vread(os.path.join(pathname, filename))
            data.append(file)

    return np.array(data, dtype=object)

def images_to_numpy(pathname, ext='.png'):
    sequence = []

    for i in range(100):
        filename = str(i) + ext
        if filename.endswith(ext):
            image = Image.open(os.path.join(pathname, filename))
            sequence.append(np.asarray(image))

    return np.array(sequence)

def load_image_sequences(pathname, ext='.png'):
    data = []

    for videoname in os.listdir(pathname):
        if os.path.isdir(os.path.join(pathname, videoname)):
            video = images_to_numpy(os.path.join(pathname, videoname), ext)
            data.append(video)

    return np.array(data)

def split_dataset(dataset, input_length=10, output_length=10, pixel_max=255.0):
    x_train = []
    y_train = []

    for v in dataset:
        v_split = np.split(v, [input_length, input_length + output_length])
        x_train.append(v_split[0])
        y_train.append(v_split[1])

    x_train = np.array(x_train, dtype='float32') / pixel_max
    y_train = np.array(y_train, dtype='float32') / pixel_max

    return x_train, y_train

def split_dataset_sequenced(dataset, input_length=10, extrapolate_length=0, pixel_max=255.0):
    x_train = []
    y_train = []

    for v in dataset:
        v_split_x = np.split(v, [input_length])
        x_train.append(v_split_x[0])

        if extrapolate_length == 0:
            v_split_y = np.split(v, [1, input_length + 1])
            y_train.append(v_split_y[1])
        else:
            v_split_y = np.split(v, [1, input_length + extrapolate_length])
            y_train.append(v_split_y[1])

    x_train = np.array(x_train, dtype='float32') / pixel_max
    y_train = np.array(y_train, dtype='float32') / pixel_max

    return x_train, y_train

def split_dataset_on_disk(dataset, filename1, filename2, input_length=10, output_length=10):
    x_train_file = np.memmap(filename1, dtype='float32', mode='r+', shape=(dataset.shape[0], input_length, dataset.shape[2], dataset.shape[3], dataset.shape[4]))
    y_train_file = np.memmap(filename2, dtype='float32', mode='r+', shape=(dataset.shape[0], output_length, dataset.shape[2], dataset.shape[3], dataset.shape[4]))
    x_train = []
    y_train = []

    for v in dataset:
        v_split = np.split(v, [input_length, input_length + output_length])
        x_train.append(v_split[0])
        y_train.append(v_split[1])

    x_train_file = np.array(x_train)[:]
    y_train_file = np.array(y_train)[:]
    del x_train
    del y_train
    gc.collect()

    return x_train_file, y_train_file

def split_dataset_by_video(pathname, input_length, output_length, target_size, pixel_max=255.0):
    Xs = np.empty((0, input_length, target_size[1], target_size[0], 3), dtype=np.uint8)
    Ys = np.empty((0, output_length, target_size[1], target_size[0], 3), dtype=np.uint8)

    for filename in os.listdir(os.path.join(pathname)):
        if filename.endswith('.npy'):
            print(f'Processing {filename}')
            v_split = np.load(os.path.join(pathname, filename), mmap_mode='r')
            v_split = list(v_split)

            while len(v_split) >= input_length + output_length:
                v_split = np.split(v_split, [input_length, input_length + output_length])

                x = np.empty((0, target_size[1], target_size[0], 3), dtype=np.uint8)
                y = np.empty((0, target_size[1], target_size[0], 3), dtype=np.uint8)

                for i in range(min(input_length, output_length)):
                    x = np.append(x, [(cv.resize(v_split[0][i] / pixel_max, target_size, interpolation=cv.INTER_LINEAR) * pixel_max).astype(np.uint8)], axis=0)
                    y = np.append(y, [(cv.resize(v_split[1][i] / pixel_max, target_size, interpolation=cv.INTER_LINEAR) * pixel_max).astype(np.uint8)], axis=0)
                if input_length > output_length:
                    for i in range(output_length, input_length):
                        x = np.append(x, [(cv.resize(v_split[0][i] / pixel_max, target_size, interpolation=cv.INTER_LINEAR) * pixel_max).astype(np.uint8)], axis=0)
                elif output_length > input_length:
                    for i in range(input_length, output_length):
                        y = np.append(y, [(cv.resize(v_split[1][i] / pixel_max, target_size, interpolation=cv.INTER_LINEAR) * pixel_max).astype(np.uint8)], axis=0)

                Xs = np.append(Xs, [x], axis=0)
                Ys = np.append(Ys, [y], axis=0)
                v_split = v_split[-1]

    return Xs, Ys

def split_dataset_by_video_sequenced(pathname, input_length, target_size, pixel_max=255.0):
    Xs = np.empty((0, input_length, target_size[1], target_size[0], 3), dtype=np.uint8)
    Ys = np.empty((0, input_length, target_size[1], target_size[0], 3), dtype=np.uint8)

    for filename in os.listdir(os.path.join(pathname)):
        if filename.endswith('.npy'):
            print(f'Processing {filename}')
            v_split = np.load(os.path.join(pathname, filename), mmap_mode='r')
            v_split = list(v_split)

            while len(v_split) >= input_length + 1:
                v_split = np.split(v_split, [input_length + 1])

                x = np.empty((0, target_size[1], target_size[0], 3), dtype=np.uint8)
                y = np.empty((0, target_size[1], target_size[0], 3), dtype=np.uint8)

                for i in range(input_length):
                    x = np.append(x, [(cv.resize(v_split[0][i] / pixel_max, target_size, interpolation=cv.INTER_LINEAR) * pixel_max).astype(np.uint8)], axis=0)
                    y = np.append(y, [(cv.resize(v_split[0][i + 1] / pixel_max, target_size, interpolation=cv.INTER_LINEAR) * pixel_max).astype(np.uint8)], axis=0)

                Xs = np.append(Xs, [x], axis=0)
                Ys = np.append(Ys, [y], axis=0)
                v_split = v_split[-1]

    return Xs, Ys

def numpy_to_hickle(pathname, target_size, pixel_max=255.0):
    Xs = np.empty((0, target_size[1], target_size[0], 3), dtype=np.uint8)
    sources = []

    for filename in os.listdir(os.path.join(pathname)):
        if filename.endswith('.npy'):
            videos = np.load(os.path.join(pathname, filename), mmap_mode='r')
            print(f'Processing {filename} with {videos.shape[0]} frames...')

            xs = np.empty((0, target_size[1], target_size[0], 3), dtype=np.uint8)
            for i in range(videos.shape[0]):
                xs = np.append(
                    xs,
                    [(cv.resize(videos[i] / pixel_max, target_size, interpolation=cv.INTER_LINEAR) * pixel_max).astype(np.uint8)],
                    axis=0)
                sources.append(filename.encode())

            Xs = np.concatenate((Xs, xs), axis=0)

    hkl.dump(np.array(Xs, dtype=np.uint8), 'X_test.hkl', mode='w', compression='gzip')
    hkl.dump(np.array(sources), 'sources_test.hkl', mode='w', compression='gzip')
    print('File size: %i bytes' % os.path.getsize('X_test.hkl'))

def imshow_frames(sequence, cmap='gray', save_to=None):
    f, ax = plt.subplots(1, sequence.shape[0], figsize=(16, 4))

    if sequence.shape[0] > 1:
        for i in range(sequence.shape[0]):
            ax[i].axis('off')
            ax[i].imshow(sequence[i], cmap=cmap)
    else:
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.axis('off')
        ax.imshow(sequence[0], cmap=cmap)

    if save_to == None:
        plt.show()
    else:
        f.savefig(save_to)
        plt.clf()

def plot_history(train_history, test_history, title, y_title, x_title='epoch', train_legend='train', test_legend='test'):
    plt.plot(train_history)
    plt.plot(test_history)
    plt.title(title)
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.legend([train_legend, test_legend], loc='best')
    plt.show()

def save_history(train_history, test_history, pathname, title, y_title, x_title='epoch', train_legend='train', test_legend='test'):
    plt.plot(train_history)
    plt.plot(test_history)
    plt.title(title)
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.legend([train_legend, test_legend], loc='best')
    plt.savefig(pathname)
    plt.clf()

def display_video(ys, y_preds, fps=25):
    vids1 = []
    vids2 = []
    for i in range(0, len(ys)):
        # Construct a GIF from the frames.
        with io.BytesIO() as gif1:
            imageio.mimsave(gif1, ys[i], "GIF", duration=(1000 / fps))
            vids1.append(gif1.getvalue())

        with io.BytesIO() as gif2:
            imageio.mimsave(gif2, y_preds[i], "GIF", duration=(1000 / fps))
            vids2.append(gif2.getvalue())

    for i in range(0, len(ys)):
        # Construct and display an `HBox` with the ground truth and prediction.
        box = HBox(
            [
                widgets.Image(value=vids1[i]),
                widgets.Image(value=vids2[i]),
            ]
        )
        display(box)