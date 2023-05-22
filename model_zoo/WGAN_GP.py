# V 0.14

import os
import time
import numpy as np
import scipy as sp
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from PIL import Image
from typing import Any
from typing import Generator


tf.random.set_seed(113)  # For reproducibility

PATH_ROOT = "training_files/"  # Root for all folders
DATASET_MEMMAP_PATH = "dataset_memmap/data.dat"  # Root for preprocessed dataset

LATENT_DIM = 128
LEARNING_RATE = 1e-4

EPOCHS = 300
BATCH_SIZE = 64

BETA_1 = 0.0
BETA_2 = 0.9

PENALTY_RATE = 10.0
CRITIC_ITER = 5


def load_data(path: str, shape: tuple = (60_000, 64, 64, 3)) -> np.ndarray:
    params = {
        "filename": os.path.join(PATH_ROOT, path),
        "dtype": "uint8",
        "mode": "r",
        "shape": shape
    }
    return (2 * np.memmap(**params).astype('float32') / 255) - 1  # Convert to range [-1, 1]


class ModelStructure:
    W_STDDEV = 0.02

    @classmethod
    def w_init(cls, shape, dtype=None) -> tf.Tensor:
        return tf.random.uniform(
            shape=shape,
            minval=-cls.W_STDDEV * np.sqrt(3),
            maxval=cls.W_STDDEV * np.sqrt(3),
            dtype=dtype
        )

    @classmethod
    def create_discriminator_model(cls) -> tf.keras.Sequential:
        model = tf.keras.Sequential()
        in_s = (64, 64, 3)

        model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=in_s,
                                use_bias=True, kernel_initializer=cls.w_init, bias_initializer='zeros'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # shape: N x 32 x 32 x 64

        model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same', use_bias=True,
                                kernel_initializer=cls.w_init, bias_initializer='zeros'))
        model.add(layers.LayerNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # shape: N x 16 x 16 x 128

        model.add(layers.Conv2D(256, kernel_size=5, strides=2, padding='same', use_bias=True,
                                kernel_initializer=cls.w_init, bias_initializer='zeros'))
        model.add(layers.LayerNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # shape: N x 8 x 8 x 256

        model.add(layers.Conv2D(512, kernel_size=5, strides=2, padding='same', use_bias=True,
                                kernel_initializer=cls.w_init, bias_initializer='zeros'))
        model.add(layers.LayerNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # shape: N x 4 x 4 x 512

        model.add(layers.Flatten())
        model.add(layers.Dense(units=1, activation='linear', use_bias=True,
                               kernel_initializer=cls.w_init, bias_initializer='zeros'))
        # shape: N x 1
        return model

    @classmethod
    def create_generator_model(cls) -> tf.keras.Sequential:
        model = tf.keras.Sequential()
        in_s = (1, 1, LATENT_DIM)
        # shape: N x 1 x 1 x LATENT_DIM

        model.add(layers.Flatten(input_shape=in_s))
        model.add(layers.Dense(units=4 * 4 * 512, activation='linear', use_bias=True,
                               kernel_initializer=cls.w_init, bias_initializer='zeros'))
        model.add(layers.Reshape((4, 4, 512)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # shape: N x 4 x 4 x 512

        model.add(layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', use_bias=True,
                                         kernel_initializer=cls.w_init, bias_initializer='zeros'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # shape: N x 8 x 8 x 256

        model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', use_bias=True,
                                         kernel_initializer=cls.w_init, bias_initializer='zeros'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # shape: N x 16 x 16 x 128

        model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=True,
                                         kernel_initializer=cls.w_init, bias_initializer='zeros'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # shape: N x 32 x 32 x 64

        model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh',
                                         use_bias=True, kernel_initializer=cls.w_init, bias_initializer='zeros'))
        # shape: N x 64 x 64 x 3
        return model


class ProgressLogger:
    PLOT_SIZE = (10, 10)
    NUM_EXAMPLES_TO_GENERATE = 25
    SEED = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, 1, 1, LATENT_DIM])

    LOG_FILENAME = "temp.txt"
    FID_FILENAME = "FID.txt"

    @classmethod
    def generate_and_save_images(cls, generator: tf.keras.Sequential, epoch: int):
        gen_images = generator(cls.SEED, training=False).numpy()
        gen_images = (255 * (gen_images + 1) / 2).astype("uint8")

        size = int(np.sqrt(cls.NUM_EXAMPLES_TO_GENERATE))
        fig, ax = plt.subplots(nrows=size, ncols=size, figsize=cls.PLOT_SIZE)
        for i in range(cls.NUM_EXAMPLES_TO_GENERATE):
            ax[i // size, i % size].axis("off")
            ax[i // size, i % size].imshow(gen_images[i])

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.05, wspace=0.05)
        plt.savefig(os.path.join(PATH_ROOT, "image_at_epoch_{:04d}.png").format(epoch))
        plt.close()

    @classmethod
    def save_and_print_log(cls, str_line: str):
        print(str_line)
        with open(os.path.join(PATH_ROOT, cls.LOG_FILENAME), "a+") as file:
            file.write(str_line + "\n")

    @classmethod
    def save_fid_log(cls, str_line: str):
        with open(os.path.join(PATH_ROOT, cls.FID_FILENAME), "a+") as file:
            file.write(str_line + "\n")


class Model:
    FREQ_SAVE = 2
    CHECKPOINT_DIR = os.path.join(PATH_ROOT, "training_checkpoints")
    CHECKPOINT_PREFIX = f"ckpt"

    INCEPTION_SIZE = (299, 299, 3)
    INCEPTION_BATCH = 3_000

    def __init__(self, learning_rate: float = 0.0001, beta_1: float = 0.0, beta_2: float = 0.9,
                 penalty_rate: float = 10.0, critic_iter: int = 5):
        self.inception_model = tf.keras.applications.InceptionV3(
            include_top=False, pooling='avg', input_shape=self.INCEPTION_SIZE)

        self.generator = ModelStructure.create_generator_model()
        self.discriminator = ModelStructure.create_discriminator_model()

        self.penalty_rate = penalty_rate
        self.critic_iter = critic_iter

        self.gen_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        self.dis_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        self.checkpoint = tf.train.Checkpoint(
            gen_optimizer=self.gen_optimizer,
            dis_optimizer=self.dis_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    @staticmethod
    def scale_images(image_array: np.ndarray, new_shape: tuple[int, int, int]) -> np.ndarray:
        image_num = image_array.shape[0]
        image_array_int = (255 * (image_array + 1) / 2).astype('uint8')
        result = np.zeros((image_num, *new_shape))
        for i in range(image_num):
            image = Image.fromarray(image_array_int[i])
            new_image = image.resize(size=new_shape[:2], resample=Image.BICUBIC)
            # noinspection PyTypeChecker
            result[i] = (2 * np.asarray(new_image, dtype='float32') / 255) - 1
        return result

    def calculate_fid(self, image_array_1: np.ndarray, image_array_2: np.ndarray) -> float:
        activation_1 = self.inception_model.predict(image_array_1, verbose=0)
        activation_2 = self.inception_model.predict(image_array_2, verbose=0)

        # Calculate mean and covariance statistics
        mu_1, sigma_1 = activation_1.mean(axis=0), np.cov(activation_1, rowvar=False)
        mu_2, sigma_2 = activation_2.mean(axis=0), np.cov(activation_2, rowvar=False)

        # Calculate sum squared difference between means
        s_diff = np.sum((mu_1 - mu_2) ** 2.0)

        # Calculate sqrt of product between cov
        cov_mean = sp.linalg.sqrtm(sigma_1.dot(sigma_2))

        # Check and correct imaginary numbers from sqrt
        if np.iscomplexobj(cov_mean):
            # noinspection PyUnresolvedReferences
            cov_mean = cov_mean.real

        # calculate score
        return s_diff + np.trace(sigma_1 + sigma_2 - 2.0 * cov_mean)

    def get_fid_score(self, dataset: np.ndarray) -> float:
        real_idx = np.random.randint(low=0, high=dataset.shape[0], size=self.INCEPTION_BATCH)
        real_img = dataset[real_idx]

        fake_seed = tf.random.normal([self.INCEPTION_BATCH, 1, 1, LATENT_DIM])
        fake_img = self.generator.predict(fake_seed, verbose=0)

        real_img_scale = self.scale_images(image_array=real_img, new_shape=self.INCEPTION_SIZE)
        fake_img_scale = self.scale_images(image_array=fake_img, new_shape=self.INCEPTION_SIZE)

        return self.calculate_fid(real_img_scale, fake_img_scale)

    def get_last_saved_epoch(self, verbose_flag: bool = True) -> int:
        epochs_offset = 0
        ckpt_path = tf.train.latest_checkpoint(self.CHECKPOINT_DIR)
        if ckpt_path:
            self.checkpoint.restore(ckpt_path)
            with open(os.path.join(self.CHECKPOINT_DIR, "checkpoint"), mode="r") as f:
                last_checkpoint_num = int(f.readline().split(":")[-1].strip()[1:-1].split("-")[-1])
            epochs_offset = self.FREQ_SAVE * last_checkpoint_num
            if verbose_flag:
                print(f"\nStart from {epochs_offset} epoch\n")
        return epochs_offset

    @staticmethod
    def _get_random_idx_batch(low: int, high: int, size: int) -> np.ndarray:
        return np.random.randint(low=low, high=high, size=size)

    def random_indexes_generator(self, dataset_size: int, batch_size: int) -> Generator[np.ndarray, Any, None]:
        for _ in range(dataset_size // batch_size):
            yield self._get_random_idx_batch(0, dataset_size, batch_size)

    @tf.function
    def train_step(self, real_image_batch: tf.Tensor, batch_size: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        noise_shape = (batch_size, 1, 1, LATENT_DIM)

        # Generator training
        with tf.GradientTape() as gen_tape:
            noise_batch = tf.random.normal(noise_shape)
            fake_image_batch = self.generator(noise_batch, training=True)

            real_score = self.discriminator(real_image_batch, training=True)
            fake_score = self.discriminator(fake_image_batch, training=True)

            # Calculate normal loss
            gen_loss = tf.reduce_mean(real_score) - tf.reduce_mean(fake_score)

        # Update generator weights
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

        # Discriminator training
        for _ in range(self.critic_iter):
            with tf.GradientTape() as dis_tape:
                noise_batch = tf.random.normal(noise_shape)
                fake_image_batch = self.generator(noise_batch, training=True)

                real_score = self.discriminator(real_image_batch, training=True)
                fake_score = self.discriminator(fake_image_batch, training=True)

                # Calculating scores mean
                real_score_mean = tf.reduce_mean(real_score)
                fake_score_mean = tf.reduce_mean(fake_score)

                # Calculate normal loss
                dis_loss = fake_score_mean - real_score_mean

                # Calculation penalty
                alpha = tf.random.uniform((batch_size, 1, 1, 1))
                interpolation = real_image_batch + alpha * (fake_image_batch - real_image_batch)
                gradients = tf.gradients(self.discriminator(interpolation), [interpolation])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=(1, 2, 3)))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

                # Correcting loss values with gradient penalty
                dis_loss += self.penalty_rate * gradient_penalty

            # Update discriminator weights each `critic` iteration
            dis_grad = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)
            self.dis_optimizer.apply_gradients(zip(dis_grad, self.discriminator.trainable_variables))

        # Return some statistics from last iterations and mean scores over last batch
        return gen_loss, dis_loss, real_score_mean, fake_score_mean

    def train(self, dataset: np.ndarray, epochs: int = 300, batch_size: int = 64):
        epochs_offset = self.get_last_saved_epoch()
        for epoch in range(epochs_offset, epochs):
            start = time.time()
            for idx in self.random_indexes_generator(dataset.shape[0], batch_size=batch_size):
                self.train_step(real_image_batch=dataset[idx], batch_size=batch_size)

            last_idx = self._get_random_idx_batch(0, dataset.shape[0], batch_size)
            gen_loss, dis_loss, real_score, fake_score = \
                self.train_step(real_image_batch=dataset[last_idx], batch_size=batch_size)

            # Save the model and calculate FID every FREQ_SAVE epochs
            if (epoch + 1) % self.FREQ_SAVE == 0:
                self.checkpoint.save(file_prefix=os.path.join(PATH_ROOT, self.CHECKPOINT_PREFIX))
                fid_score = self.get_fid_score(dataset=dataset)
                ProgressLogger.save_fid_log(str_line=f"{epoch + 1}: {fid_score}")

            ProgressLogger.save_and_print_log(
                str_line=f"Epoch: {epoch + 1} | Time: {round(time.time() - start, 3)} s | "
                         f"G. loss: {gen_loss} | D. loss: {dis_loss} | "
                         f"Real score: {real_score} | Fake score: {fake_score}")
            ProgressLogger.generate_and_save_images(generator=self.generator, epoch=epoch + 1)


def main():
    dataset = load_data(DATASET_MEMMAP_PATH)
    print("\nDataset Loaded!\n")

    model = Model(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2,
                  penalty_rate=PENALTY_RATE, critic_iter=CRITIC_ITER)
    model.train(dataset=dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()
