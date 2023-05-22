import model_zoo


MODELS = {
    "DCGAN": model_zoo.DCGAN,
    "DCGAN_WI": model_zoo.DCGAN_WI,
    "WGAN_GP": model_zoo.WGAN_GP
}

MODEL_NAME = "WGAN_GP"

SAVE_GENERATOR_PATH = f"xtr_generators/{MODEL_NAME}"
