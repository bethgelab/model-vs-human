from dataclasses import dataclass, field
from typing import List


@dataclass
class Experiment:
    """
    Experiment parameters
    """
    plotting_conditions: List = field(default_factory=list)
    xlabel: str = 'Condition'
    data_conditions: List = field(default_factory=list)

    def __post_init__(self):
        assert len(self.plotting_conditions) == len(self.data_conditions), \
            "Length of plotting conditions " + str(self.plotting_conditions) + \
            " and data conditions " + str(self.data_conditions) + " must be same"


colour_experiment = Experiment(data_conditions=["cr", "bw"],
                               plotting_conditions=["colour", "greyscale"],
                               xlabel="Colour")

contrast_experiment = Experiment(data_conditions=["c100", "c50", "c30", "c15", "c10", "c05", "c03", "c01"],
                                 plotting_conditions=["100", "50", "30", "15", "10", "5", "3", "1"],
                                 xlabel="Contrast in percent")

high_pass_experiment = Experiment(data_conditions=["inf", "3.0", "1.5", "1.0", "0.7", "0.55", "0.45", "0.4"],
                                  plotting_conditions=["inf", "3.0", "1.5", "1.0", ".7", ".55", ".45", ".4"],
                                  xlabel="Filter standard deviation")

low_pass_experiment = Experiment(data_conditions=["0", "1", "3", "5", "7", "10", "15", "40"],
                                 plotting_conditions=["0", "1", "3", "5", "7", "10", "15", "40"],
                                 xlabel="Filter standard deviation")

phase_scrambling_experiment = Experiment(data_conditions=["0", "30", "60", "90", "120", "150", "180"],
                                         plotting_conditions=["0", "30", "60", "90", "120", "150", "180"],
                                         xlabel="Phase noise width [$^\circ$]")

power_equalisation_experiment = Experiment(data_conditions=["0", "pow"],
                                           plotting_conditions=["original", "equalised"],
                                           xlabel="Power spectrum")

false_colour_experiment = Experiment(data_conditions=["True", "False"],
                                    plotting_conditions=["true", "opponent"],
                                    xlabel="Colour")

rotation_experiment = Experiment(data_conditions=["0", "90", "180", "270"],
                                 plotting_conditions=["0", "90", "180", "270"],
                                 xlabel="Rotation angle [$^\circ$]")

_eidolon_plotting_conditions = ["0", "1", "2", "3", "4", "5", "6", "7"]
_eidolon_xlabel = "$\mathregular{{Log}_2}$ of 'reach' parameter"

eidolonI_experiment = Experiment(data_conditions=["1-10-10", "2-10-10", "4-10-10", "8-10-10",
                                                  "16-10-10", "32-10-10", "64-10-10", "128-10-10"],
                                 plotting_conditions=_eidolon_plotting_conditions.copy(),
                                 xlabel=_eidolon_xlabel)

eidolonII_experiment = Experiment(data_conditions=["1-3-10", "2-3-10", "4-3-10", "8-3-10",
                                                   "16-3-10", "32-3-10", "64-3-10", "128-3-10"],
                                  plotting_conditions=_eidolon_plotting_conditions.copy(),
                                  xlabel=_eidolon_xlabel)

eidolonIII_experiment = Experiment(data_conditions=["1-0-10", "2-0-10", "4-0-10", "8-0-10",
                                                    "16-0-10", "32-0-10", "64-0-10", "128-0-10"],
                                   plotting_conditions=_eidolon_plotting_conditions.copy(),
                                   xlabel=_eidolon_xlabel)

uniform_noise_experiment = Experiment(data_conditions=["0.0", "0.03", "0.05", "0.1", "0.2", "0.35", "0.6", "0.9"],
                                      plotting_conditions=["0.0", ".03", ".05", ".1", ".2", ".35", ".6", ".9"],
                                      xlabel="Uniform noise width")


@dataclass
class DatasetExperiments:
    name: str
    experiments: [Experiment]


def get_experiments(dataset_names):
    datasets = []
    for name in dataset_names:
        name_for_experiment = name.replace("-", "_")
        if f"{name_for_experiment}_experiment" in globals():
            experiments = eval(f"{name_for_experiment}_experiment")
            experiments.name = name
            datasets.append(DatasetExperiments(name=name, experiments=[experiments]))
        else:
            datasets.append(DatasetExperiments(name=name, experiments=[]))
    return datasets
