import os
import numpy as np

from PIL import Image
from lsun.data import export_images


# Input dataset parameters
DATASET_LSUN_PATH = "dataset_lsun"
DATASET_LSUN_UNPACKED_PATH = "dataset_lsun_unpacked"

# Output memmap parameters
DATASET_MEMMAP_PATH = "dataset_memmap"
DATASET_MEMMAP_NAME = "data.dat"

OUTPUT_IMAGE_NUMBER = 60_000  # Number of images in dataset
OUTPUT_IMAGE_SIZE = 64  # W: 64px, H: 64px
OUTPUT_IMAGE_CHANNELS = 3  # 3 color (RGB)


def check_if_have_any_folders(path: str) -> bool:
    """
        Checks if a directory has any subfolders.

        Args:
            path (str): The path to the directory.

        Returns:
            bool: True if the directory contains one or more subfolders, False otherwise.

        Example:
            >>> check_if_have_any_folders('/path/to/directory')
            True
    """
    for _, folders, _ in os.walk(path):
        if len(folders):
            return True
        return False


def check_if_folder_exist_and_try_create(path: str) -> bool:
    """
        Checks if a folder exists at the given path and tries to create it if it doesn't exist.

        Args:
            path (str): The path to the folder.

        Returns:
            bool: True if the folder already exists, or if it was successfully created. False if an error
                  occurred while creating the folder.

        Example:
            >>> check_if_folder_exist_and_try_create('/path/to/folder')
            True
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        return True
    return False


def check_if_file_exist_and_try_create(path: str) -> bool:
    """
        Checks if a file exists at the given path and tries to create it if it doesn't exist.

        Args:
            path (str): The path to the file.

        Returns:
            bool: True if the file already exists, or if it was successfully created. False if an error
                  occurred while creating the file.

        Example:
            >>> check_if_file_exist_and_try_create('/path/to/file.txt')
            True
    """
    try:
        open(path, 'x').close()
    except FileExistsError:
        return True
    return False


def unpack_lsun(suffix: str = "unpacked"):
    """
        Unpacks LSUN dataset by exporting images from subfolders to a destination directory.

        Args:
            suffix (str, optional): A suffix to add to the destination directory names. Defaults to "unpacked".

        Returns:
            None

        Example:
            >>> unpack_lsun(suffix="processed")
    """
    for root, folders, _ in os.walk(DATASET_LSUN_PATH):
        for folder in folders:
            src_path = f"{root}/{folder}"
            dst_path = f"{DATASET_LSUN_UNPACKED_PATH}/{folder}_{suffix}"

            if check_if_folder_exist_and_try_create(path=dst_path):
                print(f"[INFO]: Folder `{dst_path}` already exist\n"
                      f"[INFO]: Skipping unpacking..")
                continue

            export_images(db_path=src_path, out_dir=dst_path, flat=True)


def create_memmap(dtype: str = 'uint8'):
    """
        Creates a memory-mapped file and converts images from LSUN dataset into a memmap array.

        Args:
            dtype (str, optional): The data type to be used for the memmap array. Defaults to 'uint8'.

        Returns:
            None

        Example:
            >>> create_memmap(dtype='float32')
    """
    dst_path = f"{DATASET_MEMMAP_PATH}/{DATASET_MEMMAP_NAME}"

    if check_if_file_exist_and_try_create(path=dst_path):
        print(f"[INFO]: File `{dst_path}` already exist\n"
              f"[INFO]: Skipping memmap creation..")
        return

    run_idx = 0
    image_dim = (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_CHANNELS)
    memmap_shape = (OUTPUT_IMAGE_NUMBER, * image_dim)
    memmap_array = np.memmap(dst_path, dtype=dtype, mode='w+', shape=memmap_shape)

    for root, folders, _ in os.walk(DATASET_LSUN_UNPACKED_PATH):
        for folder in folders:
            src_path = f"{root}/{folder}"
            print(f"Covert images from {src_path}")
            for _, _, files in os.walk(src_path):
                for i, file in enumerate(files):
                    with Image.open(fp=f"{src_path}/{file}") as image:
                        width, height = image.size
                        ms = min(width, height)
                        x_off, y_off = (width - ms) // 2, (height - ms) // 2
                        image = image.crop(box=(x_off, y_off, x_off + ms, y_off + ms))
                        image = image.resize(size=image_dim[:2], resample=Image.BICUBIC)

                        # noinspection PyTypeChecker
                        memmap_array[run_idx] = np.asarray(image)
                        run_idx += 1

                    if (i + 1) % 1000 == 0:
                        print('Finished', i + 1, 'images')


def main():
    # Start process of convert data to memmap
    print(f"[INFO]: Checking `{DATASET_LSUN_UNPACKED_PATH}` for any folders..")
    if check_if_have_any_folders(path=DATASET_LSUN_UNPACKED_PATH):
        # If dataset already unpacked, proceed to create memmap
        print(f"[INFO]: Creating memory map from `{DATASET_LSUN_UNPACKED_PATH}`")
        create_memmap()
    else:
        # If dataset required unpacking first
        print(f"[INFO]: Checking `{DATASET_LSUN_PATH}` for any folders..")
        if check_if_have_any_folders(path=DATASET_LSUN_PATH):
            # First dataset unpacking
            print(f"[INFO]: Unpacking `{DATASET_LSUN_PATH}` to `{DATASET_LSUN_UNPACKED_PATH}`..")
            unpack_lsun()

            # Then create memmap
            print(f"[INFO]: Creating memory map from `{DATASET_LSUN_UNPACKED_PATH}`..")
            create_memmap()
        else:
            # If no data found, check for already created memmap
            print(f"[INFO]: Checking `{DATASET_MEMMAP_PATH}` for `{DATASET_MEMMAP_NAME}` file..")
            file_path = f"{DATASET_MEMMAP_PATH}/{DATASET_MEMMAP_NAME}"
            if check_if_file_exist_and_try_create(path=file_path):
                # No action required if memmap already exist
                print(f"[INFO]: File with memory map exist. Path: `{file_path}`")
            else:
                # Ask user to download dataset first
                os.remove(file_path)  # Remove created file in result of checking
                print(f"[INFO]: Nothing was found!\n"
                      f"[INFO]: Download ArtBench-10 dataset (original size LSUN, per-style)\n"
                      f"[INFO]: Dataset link: `https://github.com/liaopeiyuan/artbench`\n"
                      f"[INFO]: And put downloaded folders (per-style) in `{DATASET_LSUN_PATH}`")

    print("[INFO]: Finished")


if __name__ == '__main__':
    main()
