'''
FibUnet: Fibonacci-infused UNet for Video Frame Prediction
A video prediction model employing Convolutional Neural Networks, Multi-layer Perceptrons coupled with a Fibonacci-infusing rule to address one-frame prediction problem.

Variables for training the model

RANDOM_SEED             : random seed for shuffling training data and initialising model weights
TAKE                    : used to control training process, if TAKE > 0, the initial weights are loaded from WEIGHTS_FILENAME
WEIGHTS_FILENAME        : file containing weights for initialising model
BATCH_SIZE              : training batch size
EPOCHS                  : number of training epochs
LEARNING_RATE           : initial learning rate
BETA                    : Adam optimiser's first exponential decay rate
CLIP_VALUE              : maximum value for clipping gradients during training
REDUCE_LR_FACTOR        : the factor for reducing learning rate when a metric has stopped improving
REDUCE_LR_PATIENCE      : the number of epochs not witnessing an improvement in metric before reducing learning rate
INPUT_LENGTH            : length of input sequences
OUTPUT_INDEX            : predicted-frame index
INPUT_SHAPE             : input shape, raw data is scaled to this size before inputting into model
EARLY_STOPPING_MONITOR  : the metric used for ReduceLROnPlateau and EarlyStopping
EARLY_STOPPING_PATIENCE : the number of epochs not witnessing an improvement in metric before the training stops
EARLY_STOPPING_MIN_DELTA: the amount of metric improvement required to keep training
DATASET_NAME            : dataset used for training
LOSS_FUNCTION_NAME      : the loss function used for calculating prediction errors (acronym)
LOSS_FUNCTION           : the loss function used for calculating prediction errors
'''

import os
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense, Flatten, Reshape, BatchNormalization
from keras.layers import GaussianNoise
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

from unet_mlp import fibonacci, DataGenerator, downsample_block, upsample_block
from utils import save_history

RANDOM_SEED = 42

TAKE = 0
WEIGHTS_FILENAME = f'unet_mlp_1.8f5_0.final.h5'

BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 1e-3
BETA = 0.9
CLIP_VALUE = 2000
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 5

INPUT_LENGTH = 21
OUTPUT_INDEX = 25
INPUT_SHAPE = (64, 64, 1)

EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 100
EARLY_STOPPING_MIN_DELTA = 1e-8

DATASET_NAME = 'movingMNIST'
LOSS_FUNCTION_NAME = 'mse'
LOSS_FUNCTION = 'mean_squared_error'


def main():
    K.clear_session()
    tf.random.set_seed(RANDOM_SEED)

    dataset_path = DATASET_NAME
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(dataset_path + '/weights'):
        os.mkdir(dataset_path + '/weights/')
    if not os.path.exists(dataset_path + '/images'):
        os.mkdir(dataset_path + '/images/')

    fpath = os.path.join(dataset_path, 'moving_mnist64.npy')
    dataset = np.load(fpath)

    # Split into train and validation sets using indexing.
    indexes = np.arange(dataset.shape[0])
    train_index = indexes[int(0.1 * dataset.shape[0]): int(0.9 * dataset.shape[0])]  # 80%
    val_index = indexes[: int(0.1 * dataset.shape[0])]  # 10%
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    print(f'UNET + MLP summary')
    kept_indices = fibonacci(INPUT_LENGTH) + [INPUT_LENGTH - 1]
    input_len = len(kept_indices)

    block0_units = 96
    block0_kernel = 5
    block1_units = 192
    block1_kernel = 3
    block2_units = 48
    block2_kernel = 5
    block3_units = 48
    block3_kernel = 5

    dense0_units = 512
    dense1_units = 1024
    dense2_units = 256

    dropout_rate = 0.4

    inputs = Input(shape=(input_len,) + INPUT_SHAPE)

    x = GaussianNoise(0.1)(inputs)
    # Encoding path
    f0, p0 = downsample_block(
        x, block0_units, block0_kernel, dropout_rate, pooling=False)
    f1, p1 = downsample_block(p0, block1_units, block1_kernel, dropout_rate)
    f2, p2 = downsample_block(
        p1, block2_units, block2_kernel, dropout_rate, pooling=False)
    f3, p3 = downsample_block(p2, block3_units, block3_kernel, dropout_rate)

    # Bottleneck
    x = Flatten()(p3)
    x = Dense(dense0_units,
              activation='relu',
              kernel_initializer=RandomNormal(
                  mean=max(-1, -8 / dense0_units), stddev=0.1),
              kernel_regularizer='l1_l2')(x)
    x = Dense(dense1_units,
              activation='relu',
              kernel_initializer=RandomNormal(
                  mean=max(-1, -8 / dense1_units), stddev=0.1),
              kernel_regularizer='l1_l2')(x)
    x = Dense(dense2_units,
              activation='relu',
              kernel_initializer=RandomNormal(
                  mean=max(-1, -8 / dense2_units), stddev=0.1),
              kernel_regularizer='l1_l2')(x)
    x = Dense(16 * 16 * block3_units,
              activation='relu',
              kernel_initializer=RandomNormal(
                  mean=max(-1, -8 / (16 * 16 * block3_units)), stddev=0.1),
              kernel_regularizer='l1_l2')(x)
    x = Reshape((16, 16, block3_units))(x)
    x = BatchNormalization()(x)

    # Tranpose function
    f0 = tf.transpose(f0, [0, 2, 3, 1, 4])
    f1 = tf.transpose(f1, [0, 2, 3, 1, 4])
    f2 = tf.transpose(f2, [0, 2, 3, 1, 4])
    f3 = tf.transpose(f3, [0, 2, 3, 1, 4])

    f0_shape = tf.shape(f0)
    f1_shape = tf.shape(f1)
    f2_shape = tf.shape(f2)
    f3_shape = tf.shape(f3)

    f0 = tf.reshape(f0, shape=(
        f0_shape[0], f0_shape[1], f0_shape[2], f0_shape[3] * f0_shape[4]))
    f1 = tf.reshape(f1, shape=(
        f1_shape[0], f1_shape[1], f1_shape[2], f1_shape[3] * f1_shape[4]))
    f2 = tf.reshape(f2, shape=(
        f2_shape[0], f2_shape[1], f2_shape[2], f2_shape[3] * f2_shape[4]))
    f3 = tf.reshape(f3, shape=(
        f3_shape[0], f3_shape[1], f3_shape[2], f3_shape[3] * f3_shape[4]))

    f0 = BatchNormalization()(f0)
    f1 = BatchNormalization()(f1)
    f2 = BatchNormalization()(f2)
    f3 = BatchNormalization()(f3)

    # Decoding path
    x = upsample_block(x, f3, block3_units, block3_kernel,
                       dropout_rate, duplicated=input_len)
    x = upsample_block(x, f2, block2_units, block2_kernel,
                       dropout_rate, duplicated=input_len, pooling=False)
    x = upsample_block(x, f1, block1_units, block1_kernel,
                       dropout_rate, duplicated=input_len)
    x = upsample_block(x, f0, block0_units, block0_kernel,
                       dropout_rate, duplicated=input_len, pooling=False)

    x = x * (1/51.0)
    x = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    x = x * 255
    x = Reshape((1,) + INPUT_SHAPE)(x)

    model = Model(inputs=inputs, outputs=x, name=f'unet_mlp_mnist')
    model.summary()

    print("==================================================")
    print("Slicing data")
    with tf.device("CPU"):
        train_generator = DataGenerator(train_dataset,
                                        batch_size=BATCH_SIZE,
                                        seq_length=INPUT_LENGTH,
                                        output_index=OUTPUT_INDEX,
                                        pixel_scale=1.0,
                                        masking=True,
                                        mask_indices=kept_indices,
                                        shuffle=True,
                                        seed=RANDOM_SEED)
        val_generator = DataGenerator(val_dataset,
                                      batch_size=BATCH_SIZE,
                                      seq_length=INPUT_LENGTH,
                                      output_index=OUTPUT_INDEX,
                                      pixel_scale=1.0,
                                      masking=True,
                                      mask_indices=kept_indices)

    print("==================================================")
    print("Training model")
    if TAKE > 0:
        print(
            f'Loading weights from {dataset_path}/weights/{WEIGHTS_FILENAME}')
        model.load_weights(dataset_path + '/weights/' + WEIGHTS_FILENAME)
    model.compile(loss=LOSS_FUNCTION,
                  optimizer=optimizers.Adam(
                      learning_rate=LEARNING_RATE, beta_1=BETA, clipvalue=CLIP_VALUE),
                  metrics=['accuracy'])

    callbacks = [ReduceLROnPlateau(monitor=EARLY_STOPPING_MONITOR,
                                   factor=REDUCE_LR_FACTOR,
                                   patience=REDUCE_LR_PATIENCE,
                                   min_delta=EARLY_STOPPING_MIN_DELTA,
                                   min_lr=LEARNING_RATE * 0.1),
                 CSVLogger(dataset_path + '/history.csv', append=True)]

    filepath = dataset_path + '/weights/' + model.name + '_' + str(TAKE)
    checkpoint = ModelCheckpoint(filepath=filepath + '.{epoch:04d}.h5',
                                 save_freq=int(8000 / BATCH_SIZE * 20),
                                 save_weights_only=True)
    callbacks.append(checkpoint)
    early_stopping = EarlyStopping(monitor=EARLY_STOPPING_MONITOR,
                                   patience=EARLY_STOPPING_PATIENCE,
                                   min_delta=EARLY_STOPPING_MIN_DELTA)
    callbacks.append(early_stopping)

    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        callbacks=callbacks,
                        use_multiprocessing=True)

    model.save_weights(filepath + '.final.h5')
    np.save(dataset_path + '/history.npy', history.history)

    save_history(history.history['loss'],
                 history.history['val_loss'],
                 dataset_path + '/images/training_loss.png',
                 'UNET + MLP trained on Moving MNIST training loss',
                 y_title=LOSS_FUNCTION_NAME.upper(),
                 x_title='Epochs',
                 train_legend='Train loss',
                 test_legend='Validation loss')


if __name__ == "__main__":
    main()
