from train import MODELS
from train import MODEL_NAME

SAVE_GENERATOR_PATH = f"xtr_generators/{MODEL_NAME}"


def main():
    # Create GAN model from training
    model = MODELS[MODEL_NAME].Model()

    # Load last checkpoint in to GAN model
    model.get_last_saved_epoch()

    # Saving Generator from training GAM model
    model.generator.save(SAVE_GENERATOR_PATH)


if __name__ == '__main__':
    main()
