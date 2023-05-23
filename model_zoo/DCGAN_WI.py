# V 0.5

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


tf.random.set_seed(113)  # Set seed value for reproducibility

PATH_ROOT = "training_files/"  # Root for all folders
DATASET_MEMMAP_PATH = "dataset_memmap/data.dat"  # Root for preprocessed dataset

#
# Hyperparameters for model
#

LATENT_DIM = 100
LEARNING_RATE = 2e-4

EPOCHS = 300
BATCH_SIZE = 128

BETA_1 = 0.5
BETA_2 = 0.999


def load_data(path: str, shape: tuple = (60_000, 64, 64, 3)) -> np.ndarray:
    """
        Loads data from a memory-mapped file and returns it as a NumPy array.

        Args:
            path (str): The path to the memory-mapped file.
            shape (tuple, optional): The desired shape of the loaded data. Defaults to (60000, 64, 64, 3).

        Returns:
            np.ndarray: The loaded data as a NumPy array, with values ranging from -1 to 1.

        Example:
            >>> load_data('data.memmap', shape=(10000, 32, 32, 3))
    """

    params = {
        "filename": os.path.join(PATH_ROOT, path),
        "dtype": "uint8",
        "mode": "r",
        "shape": shape
    }
    return (2 * np.memmap(**params).astype('float32') / 255) - 1  # Convert to range [-1, 1]


class ModelStructure:
    """
        A class that defines the structure of the discriminator and generator models for a GAN.

        Attributes:
            W_STDDEV (float): The standard deviation used for weight initialization.

        Methods:
            w_init(shape, dtype=None) -> tf.Tensor:
                Initializes weights randomly using a normal distribution.

            create_discriminator_model() -> tf.keras.Sequential:
                Creates the discriminator model.

            create_generator_model() -> tf.keras.Sequential:
                Creates the generator model.
    """

    W_STDDEV = 0.02

    @classmethod
    def w_init(cls, shape, dtype=None) -> tf.Tensor:
        """
            Initializes weights randomly using a normal distribution.

            Args:
                shape: The shape of the weight tensor.
                dtype: The data type of the weight tensor.

            Returns:
                tf.Tensor: The initialized weight tensor.
        """
        return tf.random.normal(
            shape=shape,
            mean=0,
            stddev=cls.W_STDDEV,
            dtype=dtype
        )

    @classmethod
    def create_discriminator_model(cls) -> tf.keras.Sequential:
        """
            Creates the discriminator model.

            Returns:
                tf.keras.Sequential: The discriminator model.
        """

        params = {
            "use_bias": False,
            "kernel_initializer": cls.w_init
        }
        model = tf.keras.Sequential()
        in_s = (64, 64, 3)

        model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=in_s, **params))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # shape: N x 32 x 32 x 64

        model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same', **params))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # shape: N x 16 x 16 x 128

        model.add(layers.Conv2D(256, kernel_size=5, strides=2, padding='same', **params))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # shape: N x 8 x 8 x 256

        model.add(layers.Conv2D(512, kernel_size=5, strides=2, padding='same', **params))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # shape: N x 4 x 4 x 512

        model.add(layers.Conv2D(512, kernel_size=4, strides=1, padding='valid', **params))
        # shape: N x 1 x 1 x 512

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))  # TODO: missing weight initialization))())
        return model

    @classmethod
    def create_generator_model(cls) -> tf.keras.Sequential:
        """
            Creates the generator model.

            Returns:
                tf.keras.Sequential: The generator model.
        """

        params = {
            "use_bias": False,
            "kernel_initializer": cls.w_init
        }
        model = tf.keras.Sequential()
        in_s = (1, 1, LATENT_DIM)

        model.add(layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding='valid', input_shape=in_s, **params))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # shape: N x 4 x 4 x 512

        model.add(layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', **params))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # shape: N x 8 x 8 x 256

        model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', **params))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # shape: N x 16 x 16 x 128

        model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', **params))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        # shape: N x 32 x 32 x 64

        model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh', **params))
        # shape: N x 64 x 64 x 3
        return model


class ProgressLogger:
    """
        A class that provides logging and saving functionalities for monitoring the progress of a GAN.

        Attributes:
            PLOT_SIZE (tuple): The size of the plot for generating and saving images.
            NUM_EXAMPLES_TO_GENERATE (int): The number of examples to generate and save.
            SEED (tf.Tensor): The seed for generating random images.
            LOG_FILENAME (str): The filename for saving the log.
            FID_FILENAME (str): The filename for saving the FID score log.

        Methods:
            generate_and_save_images(generator, epoch):
                Generates and saves images using the given generator model.

            save_and_print_log(str_line):
                Saves the log and prints the given string.

            save_fid_log(str_line):
                Saves the FID score log with the given string.
    """

    PLOT_SIZE = (10, 10)
    NUM_EXAMPLES_TO_GENERATE = 25
    SEED = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, 1, 1, LATENT_DIM])

    LOG_FILENAME = "temp.txt"
    FID_FILENAME = "FID.txt"

    @classmethod
    def generate_and_save_images(cls, generator: tf.keras.Sequential, epoch: int):
        """
            Generates and saves images using the given generator model.

            Args:
                generator (tf.keras.Sequential): The generator model.
                epoch (int): The current epoch number.
        """

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
        """
            Saves the log and prints the given string.

            Args:
                str_line (str): The string to be saved and printed.
        """

        print(str_line)
        with open(os.path.join(PATH_ROOT, cls.LOG_FILENAME), "a+") as file:
            file.write(str_line + "\n")

    @classmethod
    def save_fid_log(cls, str_line: str):
        """
            Saves the FID score log with the given string.

            Args:
                str_line (str): The string to be saved in the FID score log.
        """

        with open(os.path.join(PATH_ROOT, cls.FID_FILENAME), "a+") as file:
            file.write(str_line + "\n")


class Model:
    """
        A class that represents a GAN model for training and evaluation.

        Attributes:
            FREQ_SAVE (int): The frequency at which to save the model and calculate FID score.
            CHECKPOINT_DIR (str): The directory for saving checkpoints.
            CHECKPOINT_PREFIX (str): The prefix for checkpoint filenames.
            INCEPTION_SIZE (tuple): The size of the input images for InceptionV3.
            INCEPTION_BATCH (int): The batch size for calculating FID score.

        Methods:
            __init__(learning_rate, beta_1, beta_2, penalty_rate, critic_iter):
                Initializes the Model object with the specified hyperparameters.

            scale_images(image_array, new_shape) -> np.ndarray:
                Scales the given image array to the specified shape.

            calculate_fid(image_array_1, image_array_2) -> float:
                Calculates the Frechet Inception Distance (FID) score between two sets of images.

            get_fid_score(dataset) -> float:
                Calculates the FID score between real and generated images from the given dataset.

            discriminator_loss(self, real_output: np.ndarray, fake_output: np.ndarray) -> np.ndarray:
                Computes the total discriminator loss given the scores of real and fake outputs.

            generator_loss(self, fake_output: np.ndarray) -> np.ndarray:
                Computes the generator loss given the scores of the fake outputs.

            get_last_saved_epoch(verbose_flag=True) -> int:
                Returns the epoch number of the last saved model checkpoint.

            get_random_idx_batch(low: int, high: int, size: int) -> np.ndarray:
                Generates a random batch of indexes within the specified range.

            random_indexes_generator(dataset_size, batch_size) -> Generator[np.ndarray, Any, None]:
                Generates random indexes for creating mini-batches during training.

            train_step(real_image_batch, batch_size) -> tuple[np.ndarray, np.ndarray, float, float]:
                Performs a single training step on the GAN model.

            train(dataset, epochs=300, batch_size=64):
                Trains the GAN model on the given dataset for the specified number of epochs.
    """

    FREQ_SAVE = 5
    CHECKPOINT_DIR = os.path.join(PATH_ROOT, "training_checkpoints")
    CHECKPOINT_PREFIX = f"ckpt"

    INCEPTION_SIZE = (299, 299, 3)
    INCEPTION_BATCH = 3_000

    def __init__(self, learning_rate: float = 0.0001, beta_1: float = 0.5, beta_2: float = 0.999):
        """
            Initializes the Model object with the specified hyperparameters.

            Args:
                learning_rate (float): The learning rate for the optimizer.
                beta_1 (float): The beta_1 parameter for the optimizer.
                beta_2 (float): The beta_2 parameter for the optimizer.
        """

        self.inception_model = tf.keras.applications.InceptionV3(
            include_top=False, pooling='avg', input_shape=self.INCEPTION_SIZE)

        self.generator = ModelStructure.create_generator_model()
        self.discriminator = ModelStructure.create_discriminator_model()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

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
        """
            Scales the given image array to the specified shape.

            Args:
                image_array (np.ndarray): The array of images to scale.
                new_shape (tuple): The new shape of the images.

            Returns:
                np.ndarray: The scaled image array.
        """

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
        """
            Calculates the Frechet Inception Distance (FID) score between two sets of images.

            Args:
                image_array_1 (np.ndarray): The first set of images.
                image_array_2 (np.ndarray): The second set of images.

            Returns:
                float: The FID score.
        """

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
        """
            Calculates the FID score between real and generated images from the given dataset.

            Args:
                dataset (np.ndarray): The dataset containing real images.

            Returns:
                float: The FID score.
        """

        real_idx = np.random.randint(low=0, high=dataset.shape[0], size=self.INCEPTION_BATCH)
        real_img = dataset[real_idx]

        fake_seed = tf.random.normal([self.INCEPTION_BATCH, 1, 1, LATENT_DIM])
        fake_img = self.generator.predict(fake_seed, verbose=0)

        real_img_scale = self.scale_images(image_array=real_img, new_shape=self.INCEPTION_SIZE)
        fake_img_scale = self.scale_images(image_array=fake_img, new_shape=self.INCEPTION_SIZE)

        return self.calculate_fid(real_img_scale, fake_img_scale)

    def discriminator_loss(self, real_output: np.ndarray, fake_output: np.ndarray) -> np.ndarray:
        """
            Computes the total discriminator loss given the scores of real and fake outputs.

            Args:
                real_output (np.ndarray): The scores of the real outputs.
                fake_output (np.ndarray): The scores of the fake outputs.

            Returns:
                np.ndarray: The total discriminator loss.
        """

        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output: np.ndarray) -> np.ndarray:
        """
            Computes the generator loss given the scores of the fake outputs.

            Args:
                fake_output (np.ndarray): The scores of the fake outputs.

            Returns:
                np.ndarray: The generator loss.
        """
        return self.loss(tf.ones_like(fake_output), fake_output)

    def get_last_saved_epoch(self, verbose_flag: bool = True) -> int:
        """
            Returns the epoch number of the last saved model checkpoint.

            Args:
                verbose_flag (bool): Flag to print the starting epoch.

            Returns:
                int: The epoch number.
        """

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
    def get_random_idx_batch(low: int, high: int, size: int) -> np.ndarray:
        """
            Generates a random batch of indexes within the specified range.

            Args:
                low (int): The lower bound of the range (inclusive).
                high (int): The upper bound of the range (exclusive).
                size (int): The size of the batch.

            Returns:
                np.ndarray: The batch of random indexes.
        """
        return np.random.randint(low=low, high=high, size=size)

    def random_indexes_generator(self, dataset_size: int, batch_size: int) -> Generator[np.ndarray, Any, None]:
        """
            Generates random indexes for creating mini-batches during training.

            Args:
                dataset_size (int): The size of the dataset.
                batch_size (int): The batch size.

            Yields:
                np.ndarray: A batch of random indexes.
        """

        for _ in range(dataset_size // batch_size):
            yield self.get_random_idx_batch(0, dataset_size, batch_size)

    @tf.function
    def train_step(self, real_image_batch: tf.Tensor, batch_size: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
            Performs a single training step on the GAN model.

            Args:
                real_image_batch (tf.Tensor): The batch of real images.
                batch_size (int): The batch size.

            Returns:
                tuple: A tuple containing the generator loss, discriminator loss, real score, and fake score.
        """

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
        """
            Trains the GAN model on the given dataset for the specified number of epochs.

            Args:
                dataset (np.ndarray): The training dataset.
                epochs (int): The number of epochs to train (default: 300).
                batch_size (int): The batch size (default: 64).
        """

        epochs_offset = self.get_last_saved_epoch()
        for epoch in range(epochs_offset, epochs):
            start = time.time()
            for idx in self.random_indexes_generator(dataset.shape[0], batch_size=batch_size):
                self.train_step(real_image_batch=dataset[idx], batch_size=batch_size)

            last_idx = self.get_random_idx_batch(0, dataset.shape[0], batch_size)
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
    # Load preprocessed dataset in memmap format
    dataset = load_data(DATASET_MEMMAP_PATH)
    print("\nDataset Loaded!\n")

    # Create model using specific hyperparameters
    model = Model(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)

    # Start the training process
    model.train(dataset=dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()
