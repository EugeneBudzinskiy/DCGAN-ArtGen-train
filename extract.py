import tensorflow as tf
import config

tf.get_logger().setLevel("ERROR")  # Turn off warnings about incorrect checkpoint restoration


def main():
    print("[INFO]: Starting model extraction process")

    # Create GAN model from training
    print(f"[INFO]: Load pretrained model `{config.MODEL_NAME}`")
    model = config.MODELS[config.MODEL_NAME].Model()

    # Load last checkpoint in to GAN model
    print("[INFO]: Restore last checkpoint")
    model.get_last_saved_epoch()

    # Saving Generator from training GAM model
    print(f"[INFO]: Saving the model in `{config.SAVE_GENERATOR_PATH}`")
    model.generator.save(config.SAVE_GENERATOR_PATH)

    print("[INFO]: Finished")


if __name__ == '__main__':
    main()
