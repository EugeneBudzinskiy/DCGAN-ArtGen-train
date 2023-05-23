import model_zoo

# List of all possible model and their corresponding main file
MODELS = {
    "DCGAN": model_zoo.DCGAN,
    "DCGAN_WI": model_zoo.DCGAN_WI,
    "WGAN_GP": model_zoo.WGAN_GP
}

# Current model name
MODEL_NAME = "WGAN_GP"

# Path for saving pretrained model
SAVE_GENERATOR_PATH = f"xtr_generators/{MODEL_NAME}"
