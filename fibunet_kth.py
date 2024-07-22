from utils import save_history
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
from keras.activations import relu

from unet_mlp import allocate_gpu_growth, fibonacci, HklDataGenerator, downsample_block, upsample_block, PredictionCallback

RANDOM_SEED = 1

TAKE = 0
WEIGHTS_FILENAME = f'unet_mlp_1.7.1f5_0.final.h5'

BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 1e-3
BETA = 0.9
CLIP_VALUE = 2000
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 5
MIN_LR = LEARNING_RATE * 1e-2

INPUT_LENGTH = 21
OUTPUT_INDEX = 25
INPUT_SHAPE = (60, 80, 1)
MAX_PIXEL = 255.0

EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-12

DATASET_NAME = 'KTHaction'
LOSS_FUNCTION_NAME = 'mse'
LOSS_FUNCTION = 'mean_squared_error'


def main():
    K.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    # K.set_floatx('float64')
    allocate_gpu_growth()

    dataset_path = DATASET_NAME
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(dataset_path + '/weights'):
        os.mkdir(dataset_path + '/weights/')
    if not os.path.exists(dataset_path + '/images'):
        os.mkdir(dataset_path + '/images/')

    print(f'UNET + MLP summary')
    kept_indices = fibonacci(INPUT_LENGTH) + [INPUT_LENGTH - 1]
    input_len = len(kept_indices)

    block0_units = 192
    block0_kernel = 5
    block1_units = 128
    block1_kernel = 5
    block2_units = 64
    block2_kernel = 3
    block3_units = 32
    block3_kernel = 3

    dense0_units = 512
    dense1_units = 1024
    dense2_units = 512

    dropout_rate = 0.4

    inputs = Input(shape=(input_len,) + INPUT_SHAPE)

    x = GaussianNoise(1e-3 * MAX_PIXEL, dtype='float32')(inputs)
    f0, p0 = downsample_block(
        x, block0_units, block0_kernel, dropout_rate, pooling=False)
    f1, p1 = downsample_block(
        p0, block1_units, block1_kernel, dropout_rate)
    f2, p2 = downsample_block(
        p1, block2_units, block2_kernel, dropout_rate, pooling=False)
    f3, p3 = downsample_block(
        p2, block3_units, block3_kernel, dropout_rate)

    x = Flatten(dtype='float32')(p3)
    x = Dense(dense0_units,
              activation='relu',
              kernel_initializer=RandomNormal(
                  mean=max(-1, -8 / dense0_units), stddev=0.1),
              kernel_regularizer='l1_l2',
              dtype='float32')(x)
    x = Dense(dense1_units,
              activation='relu',
              kernel_initializer=RandomNormal(
                  mean=max(-1, -8 / dense1_units), stddev=0.1),
              kernel_regularizer='l1_l2',
              dtype='float32')(x)
    x = Dense(dense2_units,
              activation='relu',
              kernel_initializer=RandomNormal(
                  mean=max(-1, -8 / dense2_units), stddev=0.1),
              kernel_regularizer='l1_l2',
              dtype='float32')(x)
    x = Dense(15 * 20 * block3_units,
              activation='relu',
              kernel_initializer=RandomNormal(
                  mean=max(-1, -8 / (8 * 8 * block3_units)), stddev=0.1),  # don't change this mean
              kernel_regularizer='l1_l2',
              dtype='float32')(x)
    x = Reshape((15, 20, block3_units), dtype='float32')(x)
    x = BatchNormalization(dtype='float32')(x)

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

    # f0 = BatchNormalization()(f0)
    # f1 = BatchNormalization()(f1)
    # f2 = BatchNormalization()(f2)
    # f3 = BatchNormalization()(f3)
    f0 = GaussianNoise(1e-3 * MAX_PIXEL, dtype='float32')(f0)
    f1 = GaussianNoise(1e-3 * MAX_PIXEL, dtype='float32')(f1)
    f2 = GaussianNoise(1e-3 * MAX_PIXEL, dtype='float32')(f2)
    f3 = GaussianNoise(1e-3 * MAX_PIXEL, dtype='float32')(f3)

    x = upsample_block(x, f3, block3_units, block3_kernel,
                       dropout_rate, duplicated=input_len)
    x = upsample_block(x, f2, block2_units, block2_kernel,
                       dropout_rate, duplicated=input_len, pooling=False)
    x = upsample_block(x, f1, block1_units, block1_kernel,
                       dropout_rate, duplicated=input_len)
    x = upsample_block(x, f0, block0_units, block0_kernel,
                       dropout_rate, duplicated=input_len, pooling=False)

    x = Conv2D(INPUT_SHAPE[-1], kernel_size=1,
               padding='same', dtype='float32')(x)
    x = relu(x, max_value=MAX_PIXEL)
    x = Reshape((1,) + INPUT_SHAPE, dtype='float32')(x)

    model = Model(inputs=inputs, outputs=x, name=f'unet_mlp_kth')
    model.summary()

    print("==================================================")
    print("Slicing data")
    train_file = os.path.join(dataset_path, 'X_train_augmented.hkl')
    train_source = os.path.join(
        dataset_path, 'sources_train_augmented.hkl')
    val_file = os.path.join(dataset_path, 'X_val.hkl')
    val_sources = os.path.join(dataset_path, 'sources_val.hkl')

    with tf.device("CPU"):
        train_generator = HklDataGenerator(train_file,
                                           train_source,
                                           batch_size=BATCH_SIZE,
                                           seq_length=INPUT_LENGTH,
                                           output_index=OUTPUT_INDEX,
                                           pixel_scale=255.0 / MAX_PIXEL,
                                           masking=True,
                                           mask_indices=kept_indices,
                                           resize=INPUT_SHAPE,
                                           shuffle=True,
                                           seed=RANDOM_SEED)
        val_generator = HklDataGenerator(val_file,
                                         val_sources,
                                         batch_size=BATCH_SIZE,
                                         seq_length=INPUT_LENGTH,
                                         output_index=OUTPUT_INDEX,
                                         pixel_scale=255.0 / MAX_PIXEL,
                                         masking=True,
                                         mask_indices=kept_indices,
                                         resize=INPUT_SHAPE,
                                         unique=True)

        first_training_batch_x, first_training_batch_y = next(train_generator)

    print("==================================================")
    print("Training model")
    if TAKE > 0:
        print(
            f'Loading weights from {dataset_path}/weights/{WEIGHTS_FILENAME}')
        model.load_weights(
            dataset_path + '/weights/' + WEIGHTS_FILENAME)
    model.compile(loss=LOSS_FUNCTION,
                  optimizer=optimizers.Adam(
                      learning_rate=LEARNING_RATE, beta_1=BETA, clipvalue=CLIP_VALUE),
                  metrics=['accuracy', 'mse'])

    callbacks = [ReduceLROnPlateau(monitor=EARLY_STOPPING_MONITOR,
                                   factor=REDUCE_LR_FACTOR,
                                   patience=REDUCE_LR_PATIENCE,
                                   min_delta=EARLY_STOPPING_MIN_DELTA,
                                   min_lr=MIN_LR),
                 CSVLogger(dataset_path + '/history.csv', append=True)]

    filepath = dataset_path + \
        '/weights/' + model.name + '_' + str(TAKE)
    checkpoint = ModelCheckpoint(filepath=filepath + '.{epoch:04d}.h5',
                                 save_freq=20000,
                                 save_weights_only=True)
    callbacks.append(checkpoint)
    early_stopping = EarlyStopping(monitor=EARLY_STOPPING_MONITOR,
                                   patience=EARLY_STOPPING_PATIENCE,
                                   min_delta=EARLY_STOPPING_MIN_DELTA)
    callbacks.append(early_stopping)

    callbacks.append(PredictionCallback(model,
                                        first_training_batch_x[-1:],
                                        first_training_batch_y[-1:],
                                        save_folder=dataset_path + '/images/',
                                        dtype='float32'))

    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=2000,
                        validation_data=val_generator,
                        callbacks=callbacks)

    model.save_weights(filepath + '.final.h5')
    np.save(dataset_path + '/history.npy', history.history)

    save_history(history.history['loss'],
                 history.history['val_loss'],
                 dataset_path + '/images/training_loss.png',
                 'UNET + MLP trained on KTH Action training loss',
                 y_title=LOSS_FUNCTION_NAME.upper(),
                 x_title='Epochs',
                 train_legend='Train loss',
                 test_legend='Validation loss')


if __name__ == "__main__":
    main()
