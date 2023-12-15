import pandas as pd
import os

from .. import constants as c


def get_short_imagename(imagename):
    """Return image-specific suffix of imagename.

    This excludes a possible experiment-specific prefix,
    such as 0001_... for trial #1.
    """

    splits = imagename.split("_")
    if len(splits) > 1:
        name = splits[-2:]
        if name[0].startswith("n0"):
            # ImageNet image: keep n0... prefix
            name = name[0] + "_" + name[1]
        else:
            name = name[1]
    else:
        name = splits[0]
    return name


def read_data(path):
    """Read experimental data from csv file."""

    assert os.path.exists(path)
    df = pd.read_csv(path)
    return df


def read_all_csv_files_from_directory(dir_path):

    assert os.path.exists(dir_path)
    assert os.path.isdir(dir_path)

    df = pd.DataFrame()
    for f in sorted(os.listdir(dir_path)):
        if f.endswith(".csv"):
            df2 = read_data(os.path.join(dir_path, f))
            df2.columns = [c.lower() for c in df2.columns]
            df = pd.concat([df, df2])
    return df


def get_experimental_data(dataset, print_name=False):
    """Read all available data for an experiment."""

    if print_name:
        print(dataset.name)
    experiment_path = os.path.join(c.RAW_DATA_DIR, dataset.name)
    assert os.path.exists(experiment_path), experiment_path + " does not exist."

    df = read_all_csv_files_from_directory(experiment_path)
    df.condition = df.condition.astype(str)

    for experiment in dataset.experiments:
        if not set(experiment.data_conditions).issubset(set(df.condition.unique())):
            print(set(experiment.data_conditions))
            print(set(df.condition.unique()))
            raise ValueError("Condition mismatch")

    df = df.copy()
    df["image_id"] = df["imagename"].apply(get_short_imagename)

    return df


def crop_pdfs_in_directory(dir_path, suppress_output=True):
    """Crop all PDF plots in a directory (removing white borders),

    Args:
        dir_path: path to directory
    """
    assert os.path.exists(dir_path)
    assert os.path.isdir(dir_path)

    x = ""
    if suppress_output:
        x = " > /dev/null"

    for file in sorted(os.listdir(dir_path)):
        if file.endswith(".pdf"):
            fullpath = os.path.join(dir_path, file)
            os.system("pdfcrop " + fullpath + " " + fullpath + x)


##################################################################
# QUICK TESTS
##################################################################

assert get_short_imagename("0000_cl_s01_bw_boat_40_n02951358_5952.JPEG") == "n02951358_5952.JPEG"
assert get_short_imagename("airplane1-bicycle2.png") == "airplane1-bicycle2.png"
