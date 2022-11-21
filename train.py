import model_zoo


MODELS = {
    "DCGAN": model_zoo.DCGAN,
    "DCGAN_WI": model_zoo.DCGAN_WI,
    "WGAN_GP": model_zoo.WGAN_GP
}

MODEL_NAME = "DCGAN_WI"


def main():
    MODELS[MODEL_NAME].main()


if __name__ == '__main__':
    main()
