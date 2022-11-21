# V 0.4

import time
import numpy as np
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt


tf.random.set_seed(113)  # For reproducibility

PATH_ROOT = "training_files/"  # Root for all folders
DATASET_MEMMAP_PATH = "dataset_memmap/data.dat"

LATENT_DIM = 100
LEARNING_RATE = 2e-4

EPOCHS = 300
BATCH_SIZE = 128

BETA_1 = 0.5
BETA_2 = 0.999


def load_data(path: str, shape: tuple = (60_000, 64, 64, 3)) -> np.ndarray:
    params = {
        "filename": path,
        "dtype": "uint8",
        "mode": "r",
        "shape": shape
    }
    return (2 * np.memmap(**params).astype('float32') / 255) - 1  # Convert to range [-1, 1]


class ModelStructure:
    @classmethod
    def create_discriminator_model(cls) -> tf.keras.Sequential:
        model = tf.keras.Sequential()
        in_s = [64, 64, 3]
        # in: 64 x 64 x 3

        model.add(layers.Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False, input_shape=in_s))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # out: 32 x 32 x 64

        model.add(layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # out: 16 x 16 x 128

        model.add(layers.Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # out: 8 x 8 x 256

        model.add(layers.Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # out: 4 x 4 x 512

        model.add(layers.Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False))
        # out: 1 x 1 x 1

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    @classmethod
    def create_generator_model(cls) -> tf.keras.Sequential:
        model = tf.keras.Sequential()
        in_s = (1, 1, LATENT_DIM)
        # in: 1 x 1 x latent_dim

        model.add(
            layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding='valid', use_bias=False, input_shape=in_s))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # out: 4 x 4 x 512

        model.add(layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # out: 8 x 8 x 256

        model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # out: 16 x 16 x 128

        model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # out: 32 x 32 x 64

        model.add(
            layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh'))
        # out: 64 x 64 x 3
        return model


class ProgressLogger:
    PLOT_SIZE = (10, 10)
    NUM_EXAMPLES_TO_GENERATE = 25
    SEED = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, 1, 1, LATENT_DIM])

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
        plt.savefig(PATH_ROOT + 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    @classmethod
    def save_and_print_log(cls, str_line: str, log_filename: str = 'temp.txt'):
        print(str_line)
        with open(PATH_ROOT + log_filename, "a+") as file:
            file.write(str_line + "\n")


class Model:
    FREQ_SAVE = 5
    CHECKPOINT_DIR = PATH_ROOT + "training_checkpoints"
    CHECKPOINT_PREFIX = f"{CHECKPOINT_DIR}/ckpt"

    def __init__(self, learning_rate: float = 0.0001, beta_1: float = 0.5, beta_2: float = 0.999):
        self.generator = ModelStructure.create_generator_model()
        self.discriminator = ModelStructure.create_discriminator_model()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        self.dis_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        self.checkpoint = tf.train.Checkpoint(
            gen_optimizer=self.gen_optimizer,
            dis_optimizer=self.dis_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss(tf.ones_like(fake_output), fake_output)

    def get_last_saved_epoch(self) -> int:
        epochs_offset = 0
        ckpt_path = tf.train.latest_checkpoint(self.CHECKPOINT_DIR)
        if ckpt_path:
            self.checkpoint.restore(ckpt_path)
            with open(f"{self.CHECKPOINT_DIR}/checkpoint", "r") as f:
                last_checkpoint_num = int(f.readline().split(":")[-1].strip()[1:-1].split("-")[-1])
            epochs_offset = self.FREQ_SAVE * last_checkpoint_num
            print(f"\nStart from {epochs_offset} epoch\n")
        return epochs_offset

    @staticmethod
    def _get_random_idx_batch(low: int, high: int, size: int):
        return np.random.randint(low=low, high=high, size=size)

    def random_indexes_generator(self, dataset_size: int, batch_size: int):
        for _ in range(dataset_size // batch_size):
            yield self._get_random_idx_batch(0, dataset_size, batch_size)

    @tf.function
    def train_step(self, real_image_batch: tf.Tensor, batch_size: int):
        noise_shape = (batch_size, 1, 1, LATENT_DIM)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            noise_batch = tf.random.normal(noise_shape)
            fake_image_batch = self.generator(noise_batch, training=True)

            real_output = self.discriminator(real_image_batch, training=True)
            fake_output = self.discriminator(fake_image_batch, training=True)

            gen_loss = self.generator_loss(fake_output=fake_output)
            dis_loss = self.discriminator_loss(real_output=real_output, fake_output=fake_output)

        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        dis_grad = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.dis_optimizer.apply_gradients(zip(dis_grad, self.discriminator.trainable_variables))
        return gen_loss, dis_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)

    def train(self, dataset: np.ndarray, epochs: int = 300, batch_size: int = 64):
        epochs_offset = self.get_last_saved_epoch()
        for epoch in range(epochs_offset, epochs):
            start = time.time()
            for idx in self.random_indexes_generator(dataset.shape[0], batch_size=batch_size):
                self.train_step(real_image_batch=dataset[idx], batch_size=batch_size)

            last_idx = self._get_random_idx_batch(0, dataset.shape[0], batch_size)
            gen_loss, dis_loss, real_score, fake_score = \
                self.train_step(real_image_batch=dataset[last_idx], batch_size=batch_size)

            # Save the model every FREQ_SAVE epochs
            if (epoch + 1) % self.FREQ_SAVE == 0:
                self.checkpoint.save(file_prefix=self.CHECKPOINT_PREFIX)

            ProgressLogger.save_and_print_log(
                str_line=f"Epoch: {epoch + 1} | Time: {round(time.time() - start, 3)} s | "
                         f"G. loss: {gen_loss} | D. loss: {dis_loss} | "
                         f"Real score: {real_score} | Fake score: {fake_score}")
            ProgressLogger.generate_and_save_images(generator=self.generator, epoch=epoch + 1)


def main():
    dataset = load_data(DATASET_MEMMAP_PATH)
    print("\nDataset Loaded!\n")

    model = Model(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
    model.train(dataset=dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()
