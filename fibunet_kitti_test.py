from utils import imshow_frames, mean_psnr, mean_ssim
import os
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense, Flatten, Reshape, BatchNormalization
from keras.initializers import RandomNormal
from keras.activations import relu

from unet_mlp import fibonacci, HklDataGenerator, downsample_block, upsample_block

RANDOM_SEED = 1

TAKE = 0
WEIGHTS_FILENAME = f'unet_mlp_2.5f5_0.final.h5'

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
BETA = 0.9
CLIP_VALUE = 2000

INPUT_LENGTH = 21
OUTPUT_INDEX = 25
INPUT_SHAPE = (64, 80, 3)
MAX_PIXEL = 255.0

DATASET_NAME = 'KITTI'
LOSS_FUNCTION_NAME = 'mse'
LOSS_FUNCTION = 'mean_squared_error'


def evaluate():
    # disable Tensorflow INFO and WARNING
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    K.clear_session()
    tf.random.set_seed(RANDOM_SEED)

    dataset_path = 'CaltechPeds'
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

    f0, p0 = downsample_block(
        inputs, block0_units, block0_kernel, dropout_rate, pooling=False)
    f1, p1 = downsample_block(
        p0, block1_units, block1_kernel, dropout_rate)
    f2, p2 = downsample_block(
        p1, block2_units, block2_kernel, dropout_rate, pooling=False)
    f3, p3 = downsample_block(
        p2, block3_units, block3_kernel, dropout_rate)

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
    x = Dense(16 * 20 * block3_units,
              activation='relu',
              kernel_initializer=RandomNormal(
                  mean=max(-1, -8 / (8 * 8 * block3_units)), stddev=0.1),
              kernel_regularizer='l1_l2')(x)
    x = Reshape((16, 20, block3_units))(x)
    x = BatchNormalization()(x)

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

    x = upsample_block(x, f3, block3_units, block3_kernel,
                       dropout_rate, duplicated=input_len)
    x = upsample_block(x, f2, block2_units, block2_kernel,
                       dropout_rate, duplicated=input_len, pooling=False)
    x = upsample_block(x, f1, block1_units, block1_kernel,
                       dropout_rate, duplicated=input_len)
    x = upsample_block(x, f0, block0_units, block0_kernel,
                       dropout_rate, duplicated=input_len, pooling=False)

    x = Conv2D(INPUT_SHAPE[-1], kernel_size=1, padding='same')(x)
    x = relu(x, max_value=MAX_PIXEL)
    x = Reshape((1,) + INPUT_SHAPE)(x)

    model = Model(inputs=inputs, outputs=x, name=f'unet_mlp_kitti')
    model.summary()

    print("==================================================")
    print("Slicing data")
    test_file = os.path.join(dataset_path, 'X_test.hkl')
    test_sources = os.path.join(dataset_path, 'sources_test.hkl')

    with tf.device("CPU"):
        test_generator = HklDataGenerator(test_file,
                                          test_sources,
                                          batch_size=BATCH_SIZE,
                                          seq_length=INPUT_LENGTH,
                                          output_index=OUTPUT_INDEX,
                                          last_possible_frame=(
                                              OUTPUT_INDEX + 39),
                                          pixel_scale=255.0 / MAX_PIXEL,
                                          masking=True,
                                          mask_indices=kept_indices,
                                          resize=INPUT_SHAPE,
                                          unique=True)
        x_batch, y_batch = test_generator.next()
        x_test = np.array(x_batch)
        y_test = np.array(y_batch)
        while x_test.shape[0] <= (test_generator.N_sequences - BATCH_SIZE):
            x_batch, y_batch = test_generator.next()
            x_test = np.concatenate((x_test, x_batch), axis=0)
            y_test = np.concatenate((y_test, y_batch), axis=0)

    print("==================================================")
    print(
        f"Loading weights from {dataset_path + '/weights/' + WEIGHTS_FILENAME}")
    model.load_weights(
        dataset_path + '/weights/' + WEIGHTS_FILENAME)
    model.compile(loss=LOSS_FUNCTION,
                  optimizer=optimizers.Adam(
                      learning_rate=LEARNING_RATE, beta_1=BETA, clipvalue=CLIP_VALUE),
                  metrics=['accuracy'])

    print("==================================================")
    print(
        f"Evaluating {x_test.shape[0]}/{test_generator.N_sequences} test sequences")
    y_pred = model.predict(x_test, batch_size=BATCH_SIZE)

    x_test = x_test.astype('uint8')
    y_test = y_test.astype('uint8')
    y_pred = y_pred.astype('uint8')

    maes = np.mean(np.absolute(y_test - y_pred))
    mses = np.mean((y_test - y_pred) ** 2)
    ssims = mean_ssim(y_test, y_pred, max_val=MAX_PIXEL, filter_size=11)
    psnrs = mean_psnr(y_test, y_pred, max_val=MAX_PIXEL)

    print("MAE:", maes)
    print("MSE:", mses)
    print("SSIM: ", ssims)
    print("PSNR: ", psnrs)

    idx = 0
    for i in range(5):
        imshow_frames(x_test[idx], save_to=dataset_path +
                      f'/images/test{idx}_input.png')
        imshow_frames(y_test[idx], save_to=dataset_path +
                      f'/images/test{idx}_ground_truth.png')
        imshow_frames(y_pred[idx], save_to=dataset_path +
                      f'/images/test{idx}_output.png')
        idx = np.random.randint(x_test.shape[0])


if __name__ == "__main__":
    evaluate()
