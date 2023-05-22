import config


def main():
    # Start training process
    config.MODELS[config.MODEL_NAME].main()


if __name__ == '__main__':
    main()
