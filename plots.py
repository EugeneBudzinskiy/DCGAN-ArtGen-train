import os
import config
from matplotlib import pyplot as plt


MODEL = config.MODELS[config.MODEL_NAME]  # Get corresponding model file
MODEL_FREQ_SAVE = MODEL.Model.FREQ_SAVE  # Gen `FREQ_SAVE` from corresponding model

# Log file parameters
LOG_PATH = os.path.join(MODEL.PATH_ROOT, MODEL.ProgressLogger.LOG_FILENAME)
LOG_CELL_SEPARATOR = "|"
LOG_NAME_VALUE_SEPARATOR = ":"

# FID file parameters
FID_PATH = os.path.join(MODEL.PATH_ROOT, MODEL.ProgressLogger.FID_FILENAME)
FID_NAME_VALUE_SEPARATOR = ":"

OUTPUT_PATH = MODEL.PATH_ROOT  # Output path to save plots


def collect_log_data(path: str, cell_separator: str, name_value_separator: str) -> dict:
    """
        Collects log data from a file and organizes it into a dictionary.

        Args:
            path (str): The path to the log file.
            cell_separator (str): The separator used to split cells within each line.
            name_value_separator (str): The separator used to split name-value pairs within each cell.

        Returns:
            dict: A dictionary containing the collected log data, where the keys represent the names and the values
                  are lists of corresponding values extracted from the log file.

        Example:
            >>> collect_log_data('log.txt', ',', ':')
            {'name1': [value1, value2, ...], 'name2': [value1, value2, ...], ...}
        """
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
    """
        Saves plots based on the given data dictionary.

        Args:
            data (dict): A dictionary containing the data for plotting. The keys represent different categories of data.
                         The expected keys are "D. loss", "G. loss", "Real score", and "Fake score".
            output_path (str): The output path where the plots will be saved.

        Returns:
            None

        Example:
            >>> data_dict = {
            ...     "D. loss": ["value1", "value2", ...],
            ...     "G. loss": ["value1", "value2", ...],
            ...     "Real score": ["value1", "value2", ...],
            ...     "Fake score": ["value1", "value2", ...]
            ... }
            >>> save_log_plots(data_dict, "output/plots/")
        """
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
    """
        Collects FID (Fréchet Inception Distance) data from a file and organizes it into a dictionary.

        Args:
            path (str): The path to the file containing FID data.
            name_value_separator (str): The separator used to split the epoch-value pairs.

        Returns:
            dict: A dictionary containing the collected FID data, where the keys are "epoch" and "value". The "epoch" key
                  corresponds to a list of epoch values, and the "value" key corresponds to a list of FID values. If a value
                  is missing or empty, it is represented as None in the list.

        Example:
            >>> collect_fid_data('fid_data.txt', ':')
            {'epoch': [epoch1, epoch2, ...], 'value': [value1, value2, ...]}
        """
    with open(path, 'r') as f:
        fid_list = f.readlines()

    output = {"epoch": [], "value": []}
    for epoch_el in fid_list:
        epoch, value = map(lambda x: x.strip(), epoch_el.replace("\n", "").split(name_value_separator))
        output['epoch'].append(int(epoch))
        output['value'].append(float(value) if value else None)

    return output


def save_fid_plots(data: dict, output_path: str):
    """
        Saves a plot based on the given FID data dictionary.

        Args:
            data (dict): A dictionary containing the FID data. It should have two keys: "epoch" and "value". The "epoch" key
                         corresponds to a list of epoch values, and the "value" key corresponds to a list of FID values.
            output_path (str): The output path where the plot will be saved.

        Returns:
            None

        Example:
            >>> data_dict = {'epoch': ["epoch1", "epoch2", ...], 'value': ["value1", "value2", ...]}
            >>> save_fid_plots(data_dict, "output/plots/")
        """
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
