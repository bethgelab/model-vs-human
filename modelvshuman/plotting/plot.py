# /!usr/bin/env python3

"""
Plotting functionality
"""

import copy
import logging
import os
from os.path import join as pjoin
from multiprocessing import Process

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from . import analyses as a
from . import decision_makers as dm
from .colors import rgb
from .. import constants as consts
from ..datasets.experiments import get_experiments
from ..helper import plotting_helper as ph
from ..utils import load_dataset

logger = logging.getLogger(__name__)


##################################################################
# PLOTTING-SPECIFIC CONSTANTS AND GLOBAL SETTINGS
##################################################################


# global default boundary settings for thin gray transparent
# boundaries to avoid not being able to see the difference
# between two partially overlapping datapoints of the same color:
PLOTTING_EDGE_COLOR = (0.3, 0.3, 0.3, 0.3)
PLOTTING_EDGE_WIDTH = 0.02

METRICS = {"OOD accuracy": (a.SixteenClassAccuracy(), "16-class-accuracy"),
           "accuracy difference": (a.SixteenClassAccuracyDifference(),
                                   "16-class-accuracy-difference"),
           "observed consistency": (a.ErrorConsistency(), "observed-consistency"),
           "error consistency": (a.ErrorConsistency(), "error-consistency")}

# exclusion criteria:
# - not OOD: control condition without manipulation (e.g. 100% contrast)
# - mean human accuracy < 0.2 (error consistency etc. not meaningful)
EXCLUDE_CONDITIONS = {
    "colour": ["cr"],
    "contrast": ["c100", "c03", "c01"],
    "high-pass": ["inf", "0.55", "0.45", "0.4"],
    "low-pass": ["0", "15", "40"],
    "phase-scrambling": ["0", "150", "180"],
    "power-equalisation": ["0"],
    "false-colour": ["True"],
    "rotation": ["0"],
    "eidolonI": ["1-10-10", "64-10-10", "128-10-10"],
    "eidolonII": ["1-3-10", "32-3-10", "64-3-10", "128-3-10"],
    "eidolonIII": ["1-0-10", "16-0-10", "32-0-10", "64-0-10", "128-0-10"],
    "uniform-noise": ["0.0", "0.6", "0.9"]
}

EXCLUDE = True


##################################################################
# Main plot function to be called by user
##################################################################


def plot(plot_types,
         plotting_definition,
         dataset_names=None,
         figure_directory_name="example-figures",
         crop_PDFs=True):
    """Start different processes for different plotting types to speed up plotting."""

    processes = []
    for plot_type in plot_types:
        process = Process(target = plot_nonparallel, kwargs={"plot_types": [plot_type],
                                                             "plotting_definition": plotting_definition,
                                                             "dataset_names": dataset_names,
                                                             "figure_directory_name": figure_directory_name,
                                                             "crop_PDFs": crop_PDFs})
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    if crop_PDFs:
        ph.crop_pdfs_in_directory(os.path.join(consts.FIGURE_DIR, figure_directory_name))


def plot_nonparallel(plot_types,
                     plotting_definition,
                     dataset_names=None,
                     figure_directory_name="example-figures",
                     crop_PDFs=True):
    for plot_type in plot_types:
        assert plot_type in consts.PLOT_TYPE_TO_DATASET_MAPPING.keys(), "please select plot_types from: " + str(
            consts.PLOT_TYPE_TO_DATASET_MAPPING.keys())
        if dataset_names is not None:
            for d_name in dataset_names:
                assert d_name in consts.PLOT_TYPE_TO_DATASET_MAPPING[plot_type], f"plot_type '{plot_type}' is not available for dataset '{d_name}'. The following datasets can be plotted for this plot_type: {consts.PLOT_TYPE_TO_DATASET_MAPPING[plot_type]}"


    result_dir = os.path.join(consts.FIGURE_DIR, figure_directory_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for plot_type in plot_types:
        if dataset_names is None:
            current_dataset_names = get_dataset_names(plot_type)
        else:
            current_dataset_names = dataset_names

        datasets = get_experiments(current_dataset_names)

        if plot_type == "confusion-matrix":
            confusion_matrices_dir = os.path.join(result_dir, "confusion-matrices/")
            if not os.path.exists(confusion_matrices_dir):
                os.makedirs(confusion_matrices_dir)
            plot_confusion_matrix(datasets=datasets,
                                  decision_maker_fun=plotting_definition,
                                  result_dir=confusion_matrices_dir)
            if crop_PDFs:
                ph.crop_pdfs_in_directory(confusion_matrices_dir)

        elif plot_type == "accuracy":
            plot_accuracy(datasets=datasets,
                          decision_maker_fun=plotting_definition,
                          result_dir=result_dir)

        elif plot_type == "entropy":
            plot_entropy(datasets=datasets,
                         decision_maker_fun=plotting_definition,
                         result_dir=result_dir)

        elif plot_type == "shape-bias":
            plot_shape_bias_matrixplot(datasets=datasets,
                                       decision_maker_fun=plotting_definition,
                                       result_dir=result_dir)
            plot_shape_bias_boxplot(datasets=datasets,
                                    decision_maker_fun=plotting_definition,
                                    result_dir=result_dir)

        elif plot_type == "error-consistency":
            plot_error_consistency(datasets=datasets,
                                   decision_maker_fun=plotting_definition,
                                   result_dir=result_dir)

        elif plot_type == "benchmark-barplot":
            plot_benchmark_barplot(datasets=datasets,
                                   decision_maker_fun=plotting_definition,
                                   result_dir=result_dir)

        elif plot_type == "nonparametric-benchmark-barplot":
            for dataset_name in current_dataset_names:
                datasets = get_experiments([dataset_name])
                plot_benchmark_barplot(datasets=datasets,
                                       decision_maker_fun=plotting_definition,
                                       result_dir=result_dir,
                                       print_to_latex=False,
                                       metrics_to_plot=["OOD accuracy"],
                                       single_dataset_name=dataset_name)

        elif plot_type == "scatterplot":
            plot_scatterplot(datasets=datasets,
                             decision_maker_fun=plotting_definition,
                             result_dir=result_dir)

        elif plot_type == "error-consistency-lineplot":
            plot_error_consistency_lineplot(datasets=datasets,
                                            decision_maker_fun=plotting_definition,
                                            result_dir=result_dir)

        else:
            raise NotImplementedError("unknown plot_type: " + plot_type)


##################################################################
# Helper methods
##################################################################


def get_datasets(dataset_names, *args, **kwargs):
    dataset_list = []
    for dataset in dataset_names:
        dataset = load_dataset(dataset, *args, **kwargs)
        dataset_list.extend(dataset) if isinstance(dataset, list) else dataset_list.append(dataset)
    return dataset_list


def get_dataset_names(plot_type):
    """Given plot_type, return suitable dataset(s).

    In this regard, 'suitable' means:
    - valid plot_type <-> dataset combination
    - data is available
    """

    dataset_names = []
    dataset_candidates = consts.PLOT_TYPE_TO_DATASET_MAPPING[plot_type]

    for candidate in dataset_candidates:
        if os.path.exists(pjoin(consts.RAW_DATA_DIR, candidate)):
            dataset_names.append(candidate)

    if len(dataset_names) == 0:
        raise ValueError("No data found for the specified plot_types.")

    return dataset_names


def get_permutations(elements):
    """Return permutation of elements.

    Return value: list of tuples, where tuples are
    unique combinations of elements
    """

    permutations = []
    for i, elem1 in enumerate(elements):
        for j, elem2 in enumerate(elements):
            if i < j:
                permutations.append((elem1, elem2))
    return permutations


def exclude_conditions(dataset):
    dataset = copy.deepcopy(dataset)
    if len(dataset.experiments) > 0:
        assert dataset.name in EXCLUDE_CONDITIONS.keys()
        for c in EXCLUDE_CONDITIONS[dataset.name]:
            assert len(dataset.experiments) == 1
            assert c in dataset.experiments[0].data_conditions, f"{c} not found for {dataset.name}"
            idx = dataset.experiments[0].data_conditions.index(c)
            dataset.experiments[0].data_conditions.remove(c)
            del dataset.experiments[0].plotting_conditions[idx]
            #print(f"Dataset {dataset.name}: removing condition {c}")
    return dataset


def log(plot_type, dataset_name):
    """Print logging info for plotting to console"""

    logging_info = f"Plotting {plot_type} for dataset {dataset_name}"
    logger.info(logging_info)
    print(logging_info)


def get_human_and_CNN_subjects(subjects):
    """Split subjects into 2 lists: human, CNNs subjects."""

    assert type(subjects) is list
    human_subjects = []
    CNN_subjects = []
    for s in subjects:
        if s.startswith("subject-"):
            human_subjects.append(s)
        else:
            CNN_subjects.append(s)
    return human_subjects, CNN_subjects


def get_raw_matrix(dataset,
                   decision_maker_fun,
                   analysis,
                   value="error-consistency"):
    """Return NxN data frame of error consistencies."""

    df = ph.get_experimental_data(dataset)
    decision_makers = decision_maker_fun(df)
    subjects = dm.get_individual_decision_makers(decision_makers)

    num_subjects = len(subjects)
    matrix = np.ones([num_subjects, num_subjects])
    for i in tqdm(range(num_subjects)):
        s1 = subjects[i]
        df1 = df.loc[(df["subj"] == s1)]
        for j in range(i, num_subjects):
            s2 = subjects[j]
            df2 = df.loc[(df["subj"] == s2)]
            a = analysis.analysis(df1, df2)[value]
            matrix[i, j] = a
            matrix[j, i] = a

    plotting_names = []
    colors = []
    for s in subjects:
        attr = dm.decision_maker_to_attributes(s, decision_makers)
        if s.startswith("subject-"):
            plotting_names.append(s)
        else:
            plotting_names.append(attr["plotting_name"])
        if attr["color"] == (230 / 255.0, 230 / 255.0, 230 / 255.0):
            # supervised models: too bright for error consistency matrices
            colors.append((150 / 255.0, 150 / 255.0, 150 / 255.0))
        else:
            colors.append(attr["color"])

    assert len(colors) == matrix.shape[0] == matrix.shape[1]

    return {"matrix": pd.DataFrame(data=matrix,
                                   columns=plotting_names,
                                   index=plotting_names),
            "colors": colors}


def plotting_names_to_data_subjects(plotting_names,
                                    decision_makers):
    subjects = []
    plotting_set = set()
    for d in decision_makers:
        if d.plotting_name in plotting_names:
            subjects += d.decision_makers
            plotting_set.add(d.plotting_name)
    missing_subjects = plotting_set.symmetric_difference(set(plotting_names))
    if len(missing_subjects) > 0:
        print("Missing subjects: ")
        print(missing_subjects)
        raise ValueError("subjects missing")
    return subjects


def get_mean_over_datasets(colname,
                           metric_fun,
                           metric_name,
                           datasets,
                           decision_maker_fun):
    """Compute the mean result of metric_fun applied to datasets.

    Returns data frame with columns as follows:
    <plotting_name> (name of decision maker)
    <colname> (name of numerical column with metric results)
    <color> (plotting color)
    """

    result_df = pd.DataFrame(columns=['model', 'plotting_name', 'dataset', colname, 'color'])
    for d_orig in datasets:
        df_raw = ph.get_experimental_data(d_orig)
        if EXCLUDE:
            d = exclude_conditions(d_orig)
        else:
            d = d_orig
        for dmaker in decision_maker_fun(df_raw):
            if len(d.experiments) == 1:
                df_selection = df_raw.loc[(df_raw["subj"].isin(dmaker.decision_makers)) &
                                          (df_raw["condition"].isin(d.experiments[0].data_conditions))]
            elif len(d.experiments) == 0:
                df_selection = df_raw.loc[(df_raw["subj"].isin(dmaker.decision_makers))]
            else:
                raise ValueError("unknown")
            r1 = metric_fun.analysis(df=df_selection)
            result_df = result_df.append([{"plotting_name": dmaker.plotting_name,
                                           "dataset": d.name,
                                           colname: r1[metric_name],
                                           "color": dmaker.color}],
                                         ignore_index=True)

    # average over datasets
    result_df = result_df.groupby(['plotting_name', 'color'], as_index=False)[colname].mean()
    return result_df


##################################################################
# Individual plotting methods
##################################################################

def x_y_plot(figure_path,
             analysis,
             decision_makers,
             experiment,
             result_df):
    """Plot experimental data on an x-y plot."""

    plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(figsize=analysis.figsize)
    plt.xlabel(experiment.xlabel)
    plt.ylabel(analysis.ylabel)
    margin = analysis.ylim[1] * 0.01  # avoid cropping datapoints
    plt.ylim((analysis.ylim[0] - margin, analysis.ylim[1] + margin))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for ID in result_df["decision-maker-ID"].unique():
        decision_maker = None
        for d in decision_makers:
            if ID == d.ID:
                decision_maker = d
                break
        assert decision_maker is not None, "no matching decision maker found"
        result_list = result_df.loc[result_df["decision-maker-ID"] == ID]["yvalue"]
        plt.plot(experiment.plotting_conditions, result_list,
                 marker=decision_maker.marker, color=decision_maker.color,
                 markersize=12, linewidth=1,
                 markeredgecolor=PLOTTING_EDGE_COLOR,
                 markeredgewidth=PLOTTING_EDGE_WIDTH)

    if analysis.height_line_for_chance is not None:
        axes = plt.gca()
        x_vals = np.array(ax.get_xlim())
        y_vals = analysis.height_line_for_chance + 0 * x_vals
        plt.plot(x_vals, y_vals, ':', color="grey")

    plt.savefig(figure_path)
    plt.close()


def confusion_matrix_helper(data, output_filename,
                            plot_cbar=False, plot_labels=False):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    sns.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    sns.set(font_scale=1.4)

    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(165 / 256, 1, N)
    vals[:, 1] = np.linspace(30 / 256, 1, N)
    vals[:, 2] = np.linspace(55 / 256, 1, N)
    newcmp = ListedColormap(vals)

    ax = sns.heatmap(data, cmap=newcmp.reversed(),
                     vmin=0.0, vmax=100.0, linecolor="black", linewidths=1.0,
                     square=True, cbar=plot_cbar,
                     xticklabels=plot_labels, yticklabels=plot_labels)

    if plot_labels:
        colnames = list(data.columns)
        rownames = list(data.index.values)
        ax.set_xticklabels(colnames)
        ax.set_yticklabels(rownames)
        ax.set(xlabel="Presented category", ylabel="Decision")
    else:
        ax.set(xlabel="", ylabel="")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    sns.reset_defaults()
    sns.reset_orig()
    plt.style.use('default')


def plot_shape_bias_matrixplot(datasets,
                               decision_maker_fun,
                               result_dir,
                               analysis=a.ShapeBias(),
                               order_by='humans'):
    assert len(datasets) == 1
    ds = datasets[0]
    assert ds.name == "cue-conflict"

    log(plot_type="shape-bias-matrixplot", dataset_name=ds.name)

    df = ph.get_experimental_data(ds)

    fontsize = 25
    ticklength = 10
    markersize = 250

    classes = df["category"].unique()
    num_classes = len(classes)

    # plot setup
    fig = plt.figure(1, figsize=(12, 12), dpi=300.)
    ax = plt.gca()

    ax.set_xlim([0, 1])
    ax.set_ylim([-.5, num_classes - 0.5])

    # secondary reversed x axis
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: 1 - x, lambda x: 1 - x))

    # labels, ticks
    plt.tick_params(axis='y',
                    which='both',
                    left=False,
                    right=False,
                    labelleft=False)
    ax.set_ylabel("Shape categories", labelpad=60, fontsize=fontsize)
    ax.set_xlabel("Fraction of 'texture' decisions", fontsize=fontsize, labelpad=25)
    ax_top.set_xlabel("Fraction of 'shape' decisions", fontsize=fontsize, labelpad=25)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax_top.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.get_xaxis().set_ticks(np.arange(0, 1.1, 0.1))
    ax_top.set_ticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=fontsize, length=ticklength)
    ax_top.tick_params(axis='both', which='major', labelsize=fontsize, length=ticklength)

    # arrows on x axes
    plt.arrow(x=0, y=-1.75, dx=1, dy=0, fc='black',
              head_width=0.4, head_length=0.03, clip_on=False,
              length_includes_head=True, overhang=0.5)
    plt.arrow(x=1, y=num_classes + 0.75, dx=-1, dy=0, fc='black',
              head_width=0.4, head_length=0.03, clip_on=False,
              length_includes_head=True, overhang=0.5)

    # icons besides y axis
    ## determine order of icons
    for dmaker in decision_maker_fun(df):
        if dmaker.plotting_name == order_by:
            df_selection = df.loc[(df["subj"].isin(dmaker.decision_makers))]
            class_avgs = []
            for cl in classes:
                df_class_selection = df_selection.query("category == '{}'".format(cl))
                class_avgs.append(1 - analysis.analysis(df=df_class_selection)['shape-bias'])
            sorted_indices = np.argsort(class_avgs)
            classes = classes[sorted_indices]
            break

    # icon placement is calculated in axis coordinates
    WIDTH = 1 / num_classes  #
    XPOS = -1.25 * WIDTH  # placement left of yaxis (-WIDTH) plus some spacing (-.25*WIDTH)
    YPOS = -0.5
    HEIGHT = 1
    MARGINX = 1 / 10 * WIDTH  # vertical whitespace between icons
    MARGINY = 1 / 10 * HEIGHT  # horizontal whitespace between icons

    left = XPOS + MARGINX
    right = XPOS + WIDTH - MARGINX

    for i in range(num_classes):
        bottom = i + MARGINY + YPOS
        top = (i + 1) - MARGINY + YPOS
        iconpath = pjoin(consts.ICONS_DIR, "{}.png".format(classes[i]))
        plt.imshow(plt.imread(iconpath), extent=[left, right, bottom, top], aspect='auto', clip_on=False)

    # plot horizontal intersection lines
    for i in range(num_classes - 1):
        plt.plot([0, 1], [i + .5, i + .5], c='gray', linestyle='dotted', alpha=0.4)

    # plot average shapebias + scatter points
    for dmaker in decision_maker_fun(df):
        df_selection = df.loc[(df["subj"].isin(dmaker.decision_makers))]
        result_df = analysis.analysis(df=df_selection)
        avg = 1 - result_df['shape-bias']
        ax.plot([avg, avg], [-1, num_classes], color=dmaker.color)
        class_avgs = []
        for cl in classes:
            df_class_selection = df_selection.query("category == '{}'".format(cl))
            class_avgs.append(1 - analysis.analysis(df=df_class_selection)['shape-bias'])

        ax.scatter(class_avgs, classes,
                   color=dmaker.color,
                   marker=dmaker.marker,
                   label=dmaker.plotting_name,
                   s=markersize,
                   clip_on=False,
                   edgecolors=PLOTTING_EDGE_COLOR,
                   linewidths=PLOTTING_EDGE_WIDTH,
                   zorder=3)

    figure_path = pjoin(result_dir, f"{ds.name}_shape-bias_matrixplot.pdf")
    fig.savefig(figure_path)
    plt.close()


def plot_shape_bias_boxplot(datasets,
                            decision_maker_fun,
                            result_dir,
                            analysis=a.ShapeBias(),
                            order_by='humans'):
    assert len(datasets) == 1
    ds = datasets[0]
    assert ds.name == "cue-conflict"

    log(plot_type="shape-bias-boxplot", dataset_name=ds.name)

    df = ph.get_experimental_data(ds)

    # plot setup
    fig = plt.figure(1, figsize=(15, 4), dpi=300.)
    ax = plt.gca()
    plt.xticks(rotation=90)
    ax.set_ylabel("shape bias", fontsize=12)

    decision_maker_to_shape_bias_dict = {}
    colors = []
    labels = []
    label_colors = []
    for dmaker in decision_maker_fun(df):
        if len(dmaker.decision_makers) > 1:
            decision_maker_to_shape_bias_humans_dict = {}
            for dmaker_human in dmaker.decision_makers:
                df_selection = df.loc[(df["subj"].isin([dmaker_human]))]
                class_avgs = df_selection.groupby(["category"]).apply(lambda x: analysis.analysis(df=x)["shape-bias"])
                decision_maker_to_shape_bias_humans_dict[dmaker_human] = class_avgs.tolist()
            df_results_humans = pd.DataFrame(decision_maker_to_shape_bias_humans_dict)
            df_results_humans["humans"] = df_results_humans.mean(axis=1)

        else:
            subject_name = dmaker.decision_makers[0]
            df_selection = df.loc[(df["subj"].isin(dmaker.decision_makers))]
            class_avgs = df_selection.groupby(["category"]).apply(lambda x: analysis.analysis(df=x)["shape-bias"])
            decision_maker_to_shape_bias_dict[subject_name] = class_avgs.tolist()
        colors.append(dmaker.color)
        if subject_name in consts.TORCHVISION_MODELS:
            label_colors.append(rgb(150, 150, 150))
        else:
            label_colors.append(dmaker.color)
        labels.append(dmaker.plotting_name)

    decision_maker_to_shape_bias_dict["humans"] = df_results_humans.humans.tolist()
    df_results = pd.DataFrame(decision_maker_to_shape_bias_dict)
    boxplot = ax.boxplot(df_results,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels,
                         showfliers=False
                         )
    for element, x_axis, color, label_color in zip(boxplot["boxes"], ax.xaxis.get_ticklabels(), colors, label_colors):
        element.set(color=color)
        x_axis.set_color(label_color)
    plt.subplots_adjust(bottom=0.55)
    figure_path = pjoin(result_dir, f"{ds.name}_shape-bias_boxplot.pdf")
    fig.savefig(figure_path)
    plt.close()


def plot_error_consistency(datasets, decision_maker_fun, result_dir,
                           analysis=a.ErrorConsistency()):
    plot_matrix(datasets=datasets, analysis=analysis,
                decision_maker_fun=decision_maker_fun,
                result_dir=result_dir, plot_type="error-consistency")


def plot_matrix(datasets, analysis,
                decision_maker_fun,
                result_dir, plot_type,
                sort_by_mean=False):
    """Plot a matrix of NxN analysis values.

    Adapted from:
    https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    """

    for dataset in datasets:

        log(plot_type=plot_type, dataset_name=dataset.name)

        res = get_raw_matrix(dataset,
                             decision_maker_fun,
                             analysis)
        if sort_by_mean:
            #res = sort_matrix_by_models_mean(res)
            res = sort_matrix_by_subjects_mean(res)
            by_mean_str = "_by-mean"
        else:
            by_mean_str = ""

        colors = res["colors"]
        sns.set(font_scale=1.5)
        sns.set_style("whitegrid", {'axes.grid': False})
        f, ax = plt.subplots(figsize=(22, 18))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(res["matrix"], ax=ax, mask=None, cmap=cmap, vmax=1.0, center=0,
                              square=True, linewidths=2.0, cbar_kws={"shrink": .5},
                              xticklabels=True, yticklabels=True)

        for i, tick_label in enumerate(ax.axes.get_yticklabels()):
            tick_label.set_color(colors[i])
        for i, tick_label in enumerate(ax.axes.get_xticklabels()):
            tick_label.set_color(colors[i])

        figure_path = pjoin(result_dir,
                            f"{dataset.name}_{analysis.plotting_name.replace(' ', '-')}_matrix{by_mean_str}.pdf")
        f.savefig(figure_path, bbox_inches='tight', pad_inches=0)
        f.clear()
    plt.cla()
    plt.clf()
    plt.close()
    sns.reset_defaults()
    sns.reset_orig()
    plt.style.use('default')


def sort_matrix_by_models_mean(result_dict):
    """Helper function: Given error consistency results, sort matrix.

    The matrix will be sorted in descending order
    according to mean agreement with models.
    """

    matrix = result_dict["matrix"]
    colors = result_dict["colors"]
    # link models to colors
    models_colors_df = pd.DataFrame.from_dict({"colors": colors, "models": matrix.index})
    models_colors_df.set_index(["models"], inplace=True)

    models_only = matrix.drop(columns=[column for column in matrix.columns if column.startswith("subject-")])
    models_only["mean"] = models_only.mean(axis=1)
    models_only = models_only.sort_values(by=["mean"], ascending=False)
    name_order = models_only.index.tolist()

    # sort columns
    matrix = matrix[name_order]
    # sort rows
    matrix = matrix.reindex(name_order)
    # sort colors
    models_colors_df = models_colors_df.reindex(name_order)

    return {"matrix": matrix,
            "colors": models_colors_df.colors.tolist()}


def sort_matrix_by_subjects_mean(result_dict):
    """Helper function: Given error consistency results, sort matrix.

    The matrix will be sorted in descending order
    according to mean agreement with subjects.
    """

    matrix = result_dict["matrix"]
    colors = result_dict["colors"]
    # link models to colors
    models_colors_df = pd.DataFrame.from_dict({"colors": colors, "models": matrix.index})
    models_colors_df.set_index(["models"], inplace=True)

    humans_only = matrix[[column for column in matrix.columns if column.startswith("subject-")]]
    humans_only["mean"] = humans_only.mean(axis=1)
    models_only = humans_only.sort_values(by=["mean"], ascending=False)
    name_order = models_only.index.tolist()

    # sort columns
    matrix = matrix[name_order]
    # sort rows
    matrix = matrix.reindex(name_order)
    # sort colors
    models_colors_df = models_colors_df.reindex(name_order)

    return {"matrix": matrix,
            "colors": models_colors_df.colors.tolist()}

def plot_confusion_matrix(datasets,
                          decision_maker_fun,
                          result_dir,
                          analysis=a.ConfusionAnalysis()):
    analysis = a.ConfusionAnalysis()
    for d in datasets:

        log(plot_type="confusion-matrix", dataset_name=d.name)

        df = ph.get_experimental_data(d)

        for e in d.experiments:
            for i, cond in enumerate(e.data_conditions):
                for dmaker in decision_maker_fun(df):
                    figure_path = pjoin(result_dir,
                                        f"{analysis.plotting_name.replace(' ', '-')}_{e.name}\
                                        _{dmaker.file_name}_\
                                        {e.plotting_conditions[i]}.pdf")
                    df_selection = df.loc[(df["subj"].isin(dmaker.decision_makers)) & \
                                          (df["condition"] == cond)]
                    result_df = analysis.analysis(df=df_selection)
                    confusion_matrix_helper(output_filename=figure_path,
                                            data=result_df)


def plot_accuracy(datasets, decision_maker_fun, result_dir):
    plot_general_analyses(datasets=datasets, analysis=a.SixteenClassAccuracy(),
                          decision_maker_fun=decision_maker_fun,
                          result_dir=result_dir, plot_type="accuracy")


def plot_entropy(datasets, decision_maker_fun, result_dir):
    plot_general_analyses(datasets=datasets, analysis=a.Entropy(),
                          decision_maker_fun=decision_maker_fun,
                          result_dir=result_dir, plot_type="entropy")


def plot_error_consistency_lineplot(datasets, decision_maker_fun, result_dir):
    plot_general_analyses(datasets=datasets, analysis=a.ErrorConsistency(),
                          decision_maker_fun=decision_maker_fun,
                          result_dir=result_dir, plot_type="error-consistency-lineplot")


def plot_general_analyses(datasets, analysis, decision_maker_fun,
                          result_dir, plot_type):
    for d in datasets:
        log(plot_type=plot_type, dataset_name=d.name)

        df = ph.get_experimental_data(d)

        for e in d.experiments:
            figure_path = pjoin(result_dir,
                                f"{e.name}_{analysis.plotting_name.replace(' ', '-')}.pdf")
            decision_makers = decision_maker_fun(df)
            result_df = analysis.get_result_df(df=df,
                                               decision_makers=decision_makers,
                                               experiment=e)

            x_y_plot(figure_path=figure_path,
                     result_df=result_df,
                     decision_makers=decision_makers,
                     analysis=analysis,
                     experiment=e)


def get_raw_benchmark_df(datasets, metric_names, decision_maker_fun,
                         include_humans=True):
    def _analyse_inner(df, model, human, all_humans, dataset, colname,
                       condition):

        analysis, metric_name = METRICS[colname]

        df1 = df.loc[(df["subj"] == model)]
        df2 = df.loc[(df["subj"] == human)]
        if condition is not None:
            df1 = df1.loc[(df1["condition"] == condition)]
            df2 = df2.loc[(df2["condition"] == condition)]

        res = analysis.analysis(df1, df2)[metric_name]

        row = {'model': model,
               'human': human,
               'dataset': dataset,
               'metric': colname,
               'condition': condition,
               'value': res}
        return row

    def _analyse_outer(group_1, group_2, metric_names, dataset,
                       df, result_df):

        rows_list = []
        is_same_group = group_1 == group_2

        for i, g1 in enumerate(group_1):
            # print(i, g1)
            for colname in metric_names:

                if len(dataset.experiments) == 0:
                    experiments = ["default-experiment"]
                else:
                    experiments = dataset.experiments

                for experiment in experiments:

                    if type(experiment) is str:
                        conditions = [None]
                        dataset_name = dataset.name
                    else:
                        conditions = experiment.data_conditions
                        dataset_name = experiment.name

                    for condition in conditions:
                        for j, g2 in enumerate(group_2):
                            if is_same_group and j <= i:  # avoid 2x evaluation
                                continue
                            elif g1 == g2:  # don't compare against yourself
                                continue
                            row = _analyse_inner(df=df, model=g1, human=g2,
                                                 all_humans=sorted(group_2),
                                                 dataset=dataset_name,
                                                 colname=colname,
                                                 condition=condition)
                            rows_list.append(row)
        result_df = result_df.append(rows_list, ignore_index=True)
        return result_df

    result_df = pd.DataFrame(columns=['model', 'human', 'dataset', 'metric',
                                      'condition', 'value'])
    for dataset_orig in tqdm(datasets):
        df = ph.get_experimental_data(dataset_orig)
        decision_makers = decision_maker_fun(df)

        humans, models = dm.get_human_and_model_decision_makers(decision_makers)
        assert len(humans) > 0, "no human data found"
        assert len(models) > 0, "no model data found"

        if EXCLUDE:
            dataset = exclude_conditions(dataset_orig)
        else:
            dataset = dataset_orig

        result_df = _analyse_outer(group_1=models,
                                   group_2=humans,
                                   metric_names=metric_names,
                                   dataset=dataset,
                                   df=df, result_df=result_df)

        if include_humans:
            result_df = _analyse_outer(group_1=humans,
                                       group_2=humans,
                                       metric_names=metric_names,
                                       dataset=dataset,
                                       df=df, result_df=result_df)
    return result_df


def print_benchmark_table_accuracy_to_latex(df):
    df = copy.deepcopy(df[df["plotting_name"] != "humans"])
    df['rank'] = df['OOD accuracy'].rank(ascending=False)
    df = df.sort_values(by="rank", ascending=True)
    df = df.reset_index(drop=True)

    df = df[["plotting_name", "OOD accuracy", "rank"]]
    df.rename(columns={"plotting_name": "model",
                       "OOD accuracy": "OOD accuracy $\\uparrow$",
                       "rank": "rank $\\downarrow$"}, inplace=True)
    df["model"] = df["model"].apply(lambda x: x.replace("_", "\_"))

    # formatting cells
    formatters = dict()
    cols_bold_mapping = {"OOD accuracy $\\uparrow$": max,
                         "rank $\\downarrow$": min}

    def format_numbers(y, num_digits=3):
        return ("{:." + str(num_digits) + "f}").format(y)

    for c, func in cols_bold_mapping.items():
        m = func(df[c])
        formatters[c] = lambda y, m=m: "\\textbf{" + format_numbers(y) + "}" if y == m else format_numbers(y)

    with open(pjoin(consts.REPORT_DIR, "assets/", "benchmark_table_accuracy.tex"), 'w') as f:
        print(df.to_latex(escape=False, formatters=formatters,
                          float_format="%.3f", index=False), file=f)


def print_benchmark_table_humanlike_to_latex(df):
    df = copy.deepcopy(df[df["plotting_name"] != "humans"])
    df['acc rank'] = df['accuracy difference'].rank()
    df['err rank'] = df['error consistency'].rank(ascending=False)
    df['obs rank'] = df['observed consistency'].rank(ascending=False)
    df['mean rank'] = (df['acc rank'] + df['err rank'] + df['obs rank']) / 3.0
    df = df.sort_values(by="mean rank", ascending=True)
    df = df.reset_index(drop=True)

    df = df[["plotting_name", "accuracy difference", "observed consistency",
             "error consistency", "mean rank"]]

    df.rename(columns={"plotting_name": "model",
                       "accuracy difference": "accuracy diff. $\\downarrow$",
                       "observed consistency": "obs. consistency $\\uparrow$",
                       "error consistency": "error consistency $\\uparrow$",
                       "mean rank": "mean rank $\\downarrow$"}, inplace=True)
    df["model"] = df["model"].apply(lambda x: x.replace("_", "\_"))

    # formatting cells
    formatters = dict()
    cols_bold_mapping = {"accuracy diff. $\\downarrow$": min,
                         "obs. consistency $\\uparrow$": max,
                         "error consistency $\\uparrow$": max,
                         "mean rank $\\downarrow$": min}

    def format_numbers(y, num_digits=3):
        return ("{:." + str(num_digits) + "f}").format(y)

    for c, func in cols_bold_mapping.items():
        m = func(df[c])
        formatters[c] = lambda y, m=m: "\\textbf{" + format_numbers(y) + "}" if y == m else format_numbers(y)

    with open(pjoin(consts.REPORT_DIR, "assets/", "benchmark_table_humanlike.tex"), 'w') as f:
        print(df.to_latex(escape=False, formatters=formatters,
                          float_format="%.3f", index=False), file=f)


def plot_benchmark_barplot(datasets, decision_maker_fun, result_dir,
                           print_to_latex=True, include_humans = True,
                           metrics_to_plot=METRICS.keys(),
                           single_dataset_name=None):

    # data frame formatting; printing humanlike benchmark table to LaTeX
    metric_names = ["accuracy difference",
                    "observed consistency",
                    "error consistency"]
    df = get_raw_benchmark_df(datasets=copy.deepcopy(datasets),
                              metric_names=metric_names,
                              decision_maker_fun=decision_maker_fun,
                              include_humans=include_humans)
    decision_makers = decision_maker_fun(ph.get_experimental_data(datasets[0]))
    df_formatted = format_benchmark_df(df=df,
                                       decision_makers=decision_makers,
                                       metric_names=metric_names,
                                       include_humans=include_humans)
    if print_to_latex:
        print_benchmark_table_humanlike_to_latex(df_formatted)

    # plotting
    for colname in metrics_to_plot:

        metric_fun, metric_name = METRICS[colname]
        logging_dataset = f"and dataset {single_dataset_name}" if single_dataset_name else ""
        logging_info = f"Plotting benchmark-barplot for metric {metric_name} {logging_dataset}"
        logger.info(logging_info)
        print(logging_info)

        if metric_fun.num_input_models == 1:
            assert metric_name == "16-class-accuracy"
            df1 = get_mean_over_datasets(colname=colname,
                                         metric_fun=metric_fun,
                                         metric_name=metric_name,
                                         datasets=copy.deepcopy(datasets),
                                         decision_maker_fun=decision_maker_fun)
            if print_to_latex:
                print_benchmark_table_accuracy_to_latex(df1)
        else:
            df1 = copy.deepcopy(df_formatted)
            df1["color"] = df1["model"].apply(lambda y: dm.decision_maker_to_attributes(y, decision_makers)["color"])

        df1 = df1.sort_values(by=colname, ascending=colname != "accuracy difference")
        df1 = df1.reset_index()
        values = df1[colname]
        names = df1["plotting_name"]
        colors = df1["color"]

        humans_add_string = "" if include_humans else "_no-humans"

        if single_dataset_name:
            dataset_add_string = f"{single_dataset_name}_" if single_dataset_name else ""
            barplot(path=pjoin(result_dir, f"{single_dataset_name}_{colname.replace(' ', '-')}{humans_add_string}.pdf"),
                    names=names, values=values, colors=colors, ylabel=colname)
        else:
            barplot(path=pjoin(result_dir, f"benchmark_{metric_name.replace(' ', '-')}{humans_add_string}.pdf"),
                    names=names, values=values, colors=colors, ylabel=colname)


def format_benchmark_df(df, decision_makers, metric_names,
                        include_humans):
    ### formatting & averaging
    df["model"] = df["model"].str.replace("_", "-")
    df["dataset"] = df["dataset"].str.replace("_", "-")

    if include_humans:
        # human analysis
        dfh = df.loc[(df["model"].str.startswith('subject-', na=False))]
        # average over conditions:
        dfh = dfh.groupby(['model', 'human', 'dataset', 'metric'], as_index=False)["value"].mean()
        # average over human subjects:
        dfh = dfh.groupby(['dataset', 'metric'], as_index=False)["value"].mean()
        # average over datasets:
        dfh = dfh.groupby(['metric'], as_index=False)["value"].mean()

    # CNN-to-human analysis
    df = df.loc[(~df["model"].str.startswith('subject-', na=False))]
    # average over conditions:
    df = df.groupby(['model', 'human', 'dataset', 'metric'], as_index=False)["value"].mean()
    # average over human subjects:
    df = df.groupby(['model', 'dataset', 'metric'], as_index=False)["value"].mean()
    # average over datasets:
    df = df.groupby(['model', 'metric'], as_index=False)["value"].mean()
    # reshape to make metrics -> columns
    df = df.pivot(index='model', columns='metric', values='value').reset_index()

    if include_humans:
        human_dict = {"model": "humans"}
        for colname in metric_names:
            analysis, metric_name = METRICS[colname]
            scalar = dfh.loc[dfh["metric"] == colname]["value"].values[0]
            human_dict[colname] = scalar
        df = df.append(human_dict, ignore_index=True)

    # replace model name with the model's plotting name
    df["plotting_name"] = df["model"].apply(
        lambda y: dm.decision_maker_to_attributes(y, decision_maker_list=decision_makers)["plotting_name"])

    return df


def barplot(path, names, values, colors, ylabel=None,
            figsize=(10, 6), ylim=None):
    fig, ax = plt.subplots(figsize=figsize)
    for s in ["right", "top"]:  # , "bottom"]:
        ax.spines[s].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels([])
    if ylim is None:
        limit = np.ceil(max(values) * 10) / 10.0
        ylim = (min(0, min(values)),
                max(0, limit))
        if "benchmark_16-class-accuracy-difference" in path:
            ylim = (0, 0.15)
    plt.ylim(ylim)
    plt.ylabel(ylabel, fontsize=17)

    bar1 = plt.bar(np.arange(len(names)), values,
                   color=colors, align="center")

    for i, rect in enumerate(bar1):
        height = rect.get_height()
        width = rect.get_width()

        plot_arrow_for_models_trained_on_large_datasets = False
        if "M)" in names[i] and plot_arrow_for_models_trained_on_large_datasets:
            plt.plot(rect.get_x() + rect.get_width() / 2.0 - 0.1, rect.get_height() + 0.02 * limit,
                     marker="$\u2193$", linestyle="", color="r")
        plt.text(rect.get_x() + rect.get_width() / 2.0, 0.01 * limit,
                 names[i], ha='center',
                 va='bottom', rotation=90)

    plt.xlim([-0.8, len(names) - 1 + 0.5 * width])
    plt.savefig(path)
    plt.close()


def plot_scatterplot(datasets,
                     decision_maker_fun, result_dir,
                     metric_x=None, metric_y=None):
    """Plot scatter plot for metric_x vs. metric_y"""

    if (metric_x and metric_y is None) or (metric_x is None and metric_y):
        raise ValueError("Please specify either both or none of the metrics.")

    if metric_x and metric_y:
        assert metric_x in METRICS.keys()
        assert metric_y in METRICS.keys()

    colname = "OOD accuracy"
    analysis, metric_name = METRICS[colname]
    df_acc = get_mean_over_datasets(colname=colname,
                                    metric_fun=a.SixteenClassAccuracy(),
                                    metric_name=metric_name,
                                    datasets=copy.deepcopy(datasets),
                                    decision_maker_fun=decision_maker_fun)

    metric_names = ["accuracy difference",
                    "observed consistency",
                    "error consistency"]
    df_raw = get_raw_benchmark_df(datasets=copy.deepcopy(datasets),
                                  metric_names=metric_names,
                                  decision_maker_fun=decision_maker_fun,
                                  include_humans=True)
    decision_makers = decision_maker_fun(ph.get_experimental_data(datasets[0]))
    df_ec = format_benchmark_df(df=df_raw,
                                decision_makers=decision_makers,
                                metric_names=metric_names,
                                include_humans=True)

    df = df_acc.merge(df_ec, on="plotting_name", how="inner")
    df["marker"] = df["model"].apply(lambda y: mmarkers.MarkerStyle(
        dm.decision_maker_to_attributes(y, decision_maker_list=decision_makers)["marker"]))

    def scatterplot_log(metric_x, metric_y):
        logging_info = f"Plotting scatterplot for {metric_x} vs. {metric_y}"
        logger.info(logging_info)
        print(logging_info)

    if len(datasets) == 1:
        dataset_name = datasets[0].name
    else:
        dataset_name = "multiple-datasets"

    if metric_x and metric_y:
        scatter_plot_helper(df, metric_x, metric_y, result_dir, dataset_name)
        scatterplot_log(metric_x, metric_y)
    else:
        perm = get_permutations(METRICS.keys())
        for metric_x, metric_y in perm:
            scatter_plot_helper(df, metric_x, metric_y, result_dir, dataset_name)
            scatterplot_log(metric_x, metric_y)


def scatter_plot_helper(df, metric_x, metric_y, result_dir, dataset_name):
    fig, ax = plt.subplots(figsize=(5, 5))
    for s in ["right", "top"]:
        ax.spines[s].set_visible(False)

    for _, row in df.iterrows():
        plt.plot(row[metric_x], row[metric_y],
                 color=row["color"], marker=row["marker"],
                 markeredgecolor=PLOTTING_EDGE_COLOR,
                 markeredgewidth=PLOTTING_EDGE_WIDTH)

    # plt.xlim(left=min(0, row[metric_x]))
    # plt.xlim(left=min(df[metric_x])
    if metric_y == "error consistency" and metric_x == "OOD accuracy":
        bottom_min_val = -0.03
    else:
        bottom_min_val = 0
    plt.ylim(bottom=min(bottom_min_val, min(df[metric_y])),
             top=np.ceil(max(df[metric_y]) * 10) / 10.0)
    plt.xlabel(metric_x)
    plt.ylabel(metric_y)

    # plot dashed line
    xlim = ax.get_xlim()
    h_acc = df.loc[(df["model"] == "humans")]["OOD accuracy"].values[0]
    l1 = np.linspace(start=xlim[0], stop=xlim[1], num=100)
    l2 = np.linspace(start=xlim[0], stop=xlim[1], num=100)
    x_text = xlim[0] + (xlim[1] - xlim[0]) / 2.0
    if metric_y == "observed consistency" and metric_x == "OOD accuracy":
        for i, _ in enumerate(l1):
            l2[i] = l1[i] * h_acc + (1 - l1[i]) * (1 - h_acc)
        plt.plot(l1, l2, linestyle='dashed', color="gray", linewidth=0.8)
        y_text = x_text * h_acc + (1 - x_text) * (1 - h_acc) - 0.03
        plt.text(x=x_text, y=y_text, s="expected", color="gray", ha='center')
    elif metric_y == "error consistency" and metric_x == "OOD accuracy":
        for i, _ in enumerate(l1):
            l2[i] = 0
        plt.plot(l1, l2, linestyle='dashed', color="gray", linewidth=0.8)
        plt.text(x=x_text, y=0.01, s="expected", color="gray", ha='center')

    plt.savefig(pjoin(result_dir,
                      f"scatter-plot_{metric_x.replace(' ', '-')}_vs_{metric_y.replace(' ', '-')}_{dataset_name}.pdf"))
    plt.close()
