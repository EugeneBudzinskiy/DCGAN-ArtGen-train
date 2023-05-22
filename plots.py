import os
import config
from matplotlib import pyplot as plt

MODEL = config.MODELS[config.MODEL_NAME]
MODEL_FREQ_SAVE = MODEL.Model.FREQ_SAVE

LOG_PATH = os.path.join(MODEL.PATH_ROOT, MODEL.ProgressLogger.LOG_FILENAME)
LOG_CELL_SEPARATOR = "|"
LOG_NAME_VALUE_SEPARATOR = ":"

FID_PATH = os.path.join(MODEL.PATH_ROOT, MODEL.ProgressLogger.FID_FILENAME)
FID_NAME_VALUE_SEPARATOR = ":"

OUTPUT_PATH = MODEL.PATH_ROOT


def collect_log_data(path: str, cell_separator: str, name_value_separator: str) -> dict:
    with open(path, 'r') as f:
        log_list = f.readlines()

    output = {}
    raw_names = log_list[0].strip().split(sep=cell_separator)
    for raw_name in raw_names:
        output[raw_name.strip().split(sep=name_value_separator)[0]] = []

    for line in log_list:
        line = line.strip().split(sep=cell_separator)
        for cell in line:
            cell = cell.strip().split(sep=name_value_separator)
            try:
                value = float(cell[1].strip())
            except ValueError:
                value = cell[1].strip()
            output[cell[0]].append(value)
    return output


def save_log_plots(data: dict, output_path: str):
    mul_x = 2
    mul_y = 1
    k = 4

    plt.figure(figsize=(k * mul_x, k * mul_y))
    plt.axhline(y=0, linewidth=1, color='b', linestyle='--')
    plt.plot(data["D. loss"], label="Discriminator loss", color="tab:orange")
    plt.plot(data["G. loss"], label="Generator loss", color="tab:blue")
    plt.xlabel("epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "losses.png"))

    plt.figure(figsize=(k * mul_x, k * mul_y))
    plt.axhline(y=0, linewidth=1, color='b', linestyle='--')
    plt.plot(data["Real score"], label="Real score", color="tab:orange")
    plt.plot(data["Fake score"], label="Fake score", color="tab:blue")
    plt.xlabel("epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "scores.png"))


def collect_fid_data(path: str, name_value_separator: str):
    with open(path, 'r') as f:
        fid_list = f.readlines()

    output = {"epoch": [], "value": []}
    for epoch_el in fid_list:
        epoch, value = map(lambda x: x.strip(), epoch_el.replace("\n", "").split(name_value_separator))
        output['epoch'].append(int(epoch))
        output['value'].append(float(value) if value else None)

    return output


def save_fid_plots(data: dict, output_path: str):
    mul_x = 2
    mul_y = 1
    k = 4

    plt.figure(figsize=(k * mul_x, k * mul_y))
    plt.plot(data['epoch'], data['value'], label="FID score")
    plt.xticks(ticks=data['epoch'], labels=[int(MODEL_FREQ_SAVE * x) for x in data['epoch']])
    plt.xlabel("epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "FID.png"))


def main():
    # Saving train logs metrics
    print("[INFO]: Reading training logs..")
    log_data = collect_log_data(
        path=LOG_PATH,
        cell_separator=LOG_CELL_SEPARATOR,
        name_value_separator=LOG_NAME_VALUE_SEPARATOR
    )
    print("[INFO]: Saving plot..")
    save_log_plots(data=log_data, output_path=OUTPUT_PATH)

    # Saving train FID score
    print("[INFO]: Reading training FID..")
    fid_data = collect_fid_data(
        path=FID_PATH,
        name_value_separator=FID_NAME_VALUE_SEPARATOR
    )
    print("[INFO]: Saving plot..")
    save_fid_plots(data=fid_data, output_path=OUTPUT_PATH)

    print("[INFO]: Finished")


if __name__ == '__main__':
    main()
