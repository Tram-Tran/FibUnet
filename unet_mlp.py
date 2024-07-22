import numpy as np
import hickle as hkl
import tensorflow as tf
from keras.preprocessing.image import Iterator

from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Dropout, MaxPool2D, UpSampling2D, TimeDistributed, Concatenate, Add, ReLU
from keras.models import Model
from keras.initializers import RandomNormal
from keras.callbacks import Callback

from utils import imshow_frames


def allocate_gpu_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def fibonacci(n):
    assert str(n).isdigit() and n > 0, 'Invalid input'
    if n <= 1:
        return [1]
    else:
        indices = [1, 2]
        next_idx = indices[-1] + indices[-2]
        while next_idx <= n:
            indices.append(next_idx)
            next_idx = indices[-1] + indices[-2]

        output = [n - i for i in indices[::-1]]
        # feed actual inputs until get first Fibonacci
        output = list(range(output[0])) + output
        return output


def downsample_block(x, n_filters, kernel_size, dropout=0.0, pooling=True):
    f = TimeDistributed(Conv2D(n_filters,
                               kernel_size,
                               activation='relu',
                               padding='same',
                               kernel_initializer=RandomNormal(
                                   mean=max(-1, -8 / n_filters), stddev=0.1),
                               kernel_regularizer='l1_l2',
                               dtype='float32'))(x)
    if pooling:
        p = TimeDistributed(MaxPool2D(2, padding='same', dtype='float32'))(f)
    else:
        p = f
    p = Dropout(dropout, dtype='float32')(p)

    return f, p


def upsample_block(x, conv_features, n_filters, kernel_size, dropout=0.0, duplicated=1, pooling=True):
    x = Conv2DTranspose(n_filters,
                        kernel_size,
                        activation='relu',
                        padding='same',
                        kernel_initializer=RandomNormal(
                            mean=max(-1, -8 / n_filters), stddev=0.1),
                        kernel_regularizer='l1_l2',
                        dtype='float32')(x)
    if pooling:
        x = UpSampling2D(2, dtype='float32')(x)

    if conv_features != None:
        if duplicated > 1:
            x = tf.repeat(x, repeats=duplicated, axis=-1)
        x = Concatenate(dtype='float32')([x, conv_features])

    x = Dropout(dropout, dtype='float32')(x)
    x = Conv2D(n_filters,
               kernel_size,
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(
                   mean=max(-1, -8 / n_filters), stddev=0.1),
               kernel_regularizer='l1_l2',
               dtype='float32')(x)

    return x


def residual_block(input_shape, filters, kernel_size, kernel_regularizer=None):
    input = Input(shape=input_shape)
    # build x with the same number of filters
    x = Conv2D(filters,
               kernel_size=1,
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(
                   mean=max(-1, -8 / filters), stddev=0.1),
               kernel_regularizer=kernel_regularizer)(input)

    # build f(x) with the same number of filters
    fx = Conv2D(filters,
                kernel_size,
                activation='relu',
                padding='same',
                kernel_initializer=RandomNormal(
                    mean=max(-1, -8 / filters), stddev=0.1),
                kernel_regularizer=kernel_regularizer)(input)
    fx = Conv2D(filters,
                kernel_size,
                activation='relu',
                padding='same',
                kernel_initializer=RandomNormal(
                    mean=max(-1, -8 / filters), stddev=0.1),
                kernel_regularizer=kernel_regularizer)(fx)
    output = Add()([fx, x])
    output = ReLU()(output)

    return Model(input, output)


def downsample_res_block(x, n_filters, kernel_size, dropout=0.0, pooling=True):
    f = TimeDistributed(residual_block(input_shape=x.shape[2:],
                                       filters=n_filters,
                                       kernel_size=kernel_size,
                                       kernel_regularizer='l1_l2'))(x)
    if pooling:
        p = TimeDistributed(MaxPool2D(2, padding='same'))(f)
    else:
        p = f
    p = Dropout(dropout)(p)

    return f, p


def upsample_res_block(x, conv_features, n_filters, kernel_size, dropout=0.0, duplicated=1, pooling=True):
    x = Conv2DTranspose(n_filters,
                        kernel_size,
                        activation='relu',
                        padding='same',
                        kernel_initializer=RandomNormal(
                            mean=max(-1, -8 / n_filters), stddev=0.1),
                        kernel_regularizer='l1_l2')(x)
    if pooling:
        x = UpSampling2D(2)(x)

    if conv_features != None:
        if duplicated > 1:
            x = tf.repeat(x, repeats=duplicated, axis=-1)
        x = Concatenate()([x, conv_features])

    x = residual_block(input_shape=x.shape[1:],
                       filters=n_filters,
                       kernel_size=kernel_size,
                       kernel_regularizer='l1_l2')(x)
    x = Dropout(dropout)(x)

    return x

# Data generator that creates sequences for input into Unet.


class DataGenerator(Iterator):
    def __init__(self,
                 dataset,
                 batch_size=32,
                 seq_length=25,
                 output_index=25,
                 n_channels=1,
                 pixel_scale=255.0,
                 masking=False,
                 mask_indices=None,
                 shuffle=False,
                 seed=None,
                 dtype=np.float32):
        self.X = dataset  # numpy array
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seq_shape = self.X[0][0].shape
        if n_channels > 1:
            self.seq_shape = self.X[0][0].shape[:2] + (n_channels,)
            self.X = np.repeat(self.X, n_channels, axis=-1)
        self.output_index = output_index
        self.pixel_scale = pixel_scale

        self.masking = masking
        self.mask_indices = mask_indices

        self.dtype = dtype

        self.N_sequences = self.X.shape[0]
        super(DataGenerator, self).__init__(
            self.N_sequences, batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(
                self.index_generator), self.batch_size

        input_length = len(
            self.mask_indices) if self.masking else self.seq_length

        batch_x = np.zeros(
            (current_batch_size, input_length) + self.seq_shape, self.dtype)
        batch_y = np.zeros((current_batch_size, 1) +
                           self.seq_shape, self.dtype)

        for i, idx in enumerate(index_array):
            batch_x[i] = self.preprocess(
                self.X[idx][:self.seq_length], masking=self.masking)
            batch_y[i] = self.preprocess(self.X[idx][self.output_index])

        batch_y = np.max(batch_y, axis=-1, keepdims=True)

        return batch_x, batch_y

    def preprocess(self, X, masking=False):
        if masking:
            X = X[self.mask_indices]

        return X / self.pixel_scale

    def create_all(self):
        Xs = self.X[:, :self.seq_length, ...].astype(self.dtype)
        if self.masking:
            Xs = self.X[:, self.mask_indices, ...].astype(self.dtype)

        Ys = self.X[:, self.output_index, ...].astype(self.dtype)
        Ys = np.max(Ys, axis=-1, keepdims=True)
        Ys = np.expand_dims(Ys, axis=1)

        return (Xs / self.pixel_scale), (Ys / self.pixel_scale)

# Data generator that creates sequences from KITTI for input into Unet.


class HklDataGenerator(Iterator):
    def __init__(self,
                 data_file,
                 source_file,
                 batch_size=32,
                 seq_length=25,
                 output_index=25,
                 pixel_scale=255.0,
                 last_possible_frame=-1,
                 resize=None,
                 masking=False,
                 mask_indices=None,
                 n_seq=None,
                 unique=False,
                 shuffle=False,
                 seed=None,
                 dtype=np.float32):
        print(f'Loading {data_file}')
        # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.X = hkl.load(data_file)
        print(f'Loading {source_file}')
        # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.sources = hkl.load(source_file)

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.img_shape = self.X[0].shape if resize == None else resize
        self.output_index = output_index
        self.pixel_scale = pixel_scale
        self.resize = resize

        self.masking = masking
        self.mask_indices = mask_indices

        self.unique = unique
        self.dtype = dtype

        last_possible_frame = last_possible_frame if last_possible_frame > 0 else self.output_index
        last_possible_start = self.X.shape[0] - last_possible_frame
        curr_location = 0
        possible_starts = []
        while curr_location < last_possible_start:
            last_frame = curr_location + last_possible_frame
            if self.sources[curr_location] == self.sources[last_frame]:
                possible_starts.append(curr_location)
                if self.unique:
                    curr_location += (self.seq_length - 1)
            curr_location += 1
        self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)

        if n_seq != None:
            self.possible_starts = self.possible_starts[:n_seq]
            self.N_sequences = n_seq
        else:
            self.N_sequences = len(self.possible_starts)
        super(HklDataGenerator, self).__init__(
            self.N_sequences, batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(
                self.index_generator), self.batch_size

        input_length = len(
            self.mask_indices) if self.masking else self.seq_length

        batch_x = np.zeros(
            (current_batch_size, input_length) + self.img_shape, self.dtype)
        batch_y = np.zeros((current_batch_size, 1) +
                           self.img_shape, self.dtype)

        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(
                self.X[idx:(idx + self.seq_length)], masking=self.masking)
            batch_y[i] = self.preprocess(np.expand_dims(
                self.X[idx + self.output_index], axis=0))

        return batch_x, batch_y

    def preprocess(self, X, masking=False):
        if masking:
            X = X[self.mask_indices]
        if self.resize != None:
            Xs = np.zeros((X.shape[0],) + self.img_shape, self.dtype)
            for i in range(X.shape[0]):
                Xs[i] = tf.image.resize(X[i], size=(
                    self.resize[0], self.resize[1]))

            return Xs / self.pixel_scale
        else:
            return X / self.pixel_scale

# Callback for printing predictions


class PredictionCallback(Callback):
    def __init__(self, model, x_test, y_test, save_folder, cmap='gray', save_freq_epoch=10, dtype='uint8'):
        super().__init__()

        self.model = model
        self.x_test = x_test
        self.y_test = y_test

        self.save_folder = save_folder
        self.cmap = cmap
        self.save_freq_epoch = save_freq_epoch
        self.dtype = dtype

        imshow_frames(self.x_test[0].astype(self.dtype),
                      cmap=self.cmap,
                      save_to=self.save_folder + 'train_input.png')
        imshow_frames(self.y_test[0].astype(self.dtype),
                      cmap=self.cmap,
                      save_to=self.save_folder + 'train_ground_truth.png')

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq_epoch == 0:
            y_pred = self.model.predict(self.x_test).astype(self.dtype)
            imshow_frames(y_pred[0], cmap=self.cmap, save_to=self.save_folder +
                          f'train_output_e{epoch}.png')
