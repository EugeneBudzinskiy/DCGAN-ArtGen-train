import os
import numpy as np

from PIL import Image
from lsun.data import export_images


DATASET_LSUN_PATH = "dataset_lsun"
DATASET_LSUN_UNPACKED_PATH = "dataset_lsun_unpacked"

DATASET_MEMMAP_PATH = "dataset_memmap"
DATASET_MEMMAP_NAME = "data.dat"

OUTPUT_IMAGE_NUMBER = 60_000
OUTPUT_IMAGE_SIZE = 64  # W: 64px, H: 64px
OUTPUT_IMAGE_CHANNELS = 3  # 3 color (RGB)


def check_if_have_any_folders(path: str) -> bool:
    for _, folders, _ in os.walk(path):
        if len(folders):
            return True
        return False


def check_if_folder_exist_and_try_create(path: str) -> bool:
    try:
        os.mkdir(path)
    except FileExistsError:
        return True
    return False


def check_if_file_exist_and_try_create(path: str) -> bool:
    try:
        open(path, 'x').close()
    except FileExistsError:
        return True
    return False


def unpack_lsun(suffix: str = "unpacked"):
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
    print(f"[INFO]: Checking `{DATASET_LSUN_UNPACKED_PATH}` for any folders..")
    if check_if_have_any_folders(path=DATASET_LSUN_UNPACKED_PATH):
        print(f"[INFO]: Creating memory map from `{DATASET_LSUN_UNPACKED_PATH}`")
        create_memmap()
    else:
        print(f"[INFO]: Checking `{DATASET_LSUN_PATH}` for any folders..")
        if check_if_have_any_folders(path=DATASET_LSUN_PATH):
            print(f"[INFO]: Unpacking `{DATASET_LSUN_PATH}` to `{DATASET_LSUN_UNPACKED_PATH}`..")
            unpack_lsun()

            print(f"[INFO]: Creating memory map from `{DATASET_LSUN_UNPACKED_PATH}`..")
            create_memmap()
        else:
            print(f"[INFO]: Checking `{DATASET_MEMMAP_PATH}` for `{DATASET_MEMMAP_NAME}` file..")
            file_path = f"{DATASET_MEMMAP_PATH}/{DATASET_MEMMAP_NAME}"
            if check_if_file_exist_and_try_create(path=file_path):
                print(f"[INFO]: File with memory map exist. Path: `{file_path}`")
            else:
                os.remove(file_path)  # Remove created file in result of checking
                print(f"[INFO]: Nothing was found!\n"
                      f"[INFO]: Download ArtBench-10 dataset (original size LSUN, per-style)\n"
                      f"[INFO]: Dataset link: `https://github.com/liaopeiyuan/artbench`\n"
                      f"[INFO]: And put downloaded folders (per-style) in `{DATASET_LSUN_PATH}`")

    print("[INFO]: Finished")


if __name__ == '__main__':
    main()
