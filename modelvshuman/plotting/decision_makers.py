# /!usr/bin/env python3

"""
Define decision makers (either human participants or CNN models).
"""

from dataclasses import dataclass
import fnmatch
from matplotlib.lines import Line2D

from .. import constants as c
from .colors import *


@dataclass
class DecisionMaker:
    name_pattern: any
    df: any
    color: any = "grey"
    marker: str = "o"
    plotting_name: str = None
    file_name: str = None # name to use when saving figures

    def __post_init__(self):

        assert len(self.df.subj.unique()) > 0, "no 'subj' column found in data frame"

        if self.plotting_name is None:
            if type(self.name_pattern) is str and self.name_pattern in self.df.subj.unique():
                self.plotting_name = self.name_pattern
            else:
                self.plotting_name = "aggregated-subjects"
        else:
            self.plotting_name = self.plotting_name

        if type(self.name_pattern) is str:
            self.name_pattern = [self.name_pattern]
        else:
            assert type(self.name_pattern) is list, "type(name_pattern) needs \
                                                     to be 'str' or 'list'"
            self.name_pattern = self.name_pattern

        self.decision_makers = []
        for subj in self.df.subj.unique():
            for pattern in self.name_pattern:
                if fnmatch.fnmatch(subj, pattern):
                    self.decision_makers.append(subj)
        if len(self.decision_makers) == 0:
            print("The following model(s) were not found: "+', '.join(self.name_pattern))
            print("List of possible models in this dataset:")
            print(self.df.subj.unique())
        else:
            self.ID = self._get_ID()
        self.df = None # no longer needed

        assert self.marker in Line2D.markers.keys(), "Unknown marker "+str(self.marker)+" used when creating decision maker: use one of "+str(Line2D.markers.keys())

        self.file_name = self._convert_file_name(self.plotting_name)


    def _get_ID(self):
        ID = self.decision_makers[0]
        for d in self.decision_makers[1:]:
            ID = ID+"_"+d
        return ID

    def _convert_file_name(self, plotting_name):
        file_name = plotting_name
        file_name = file_name.replace(" ", "")
        file_name = file_name.replace("_", "-")
        file_name = file_name.replace(",", "-")
        return file_name


def get_individual_decision_makers(decision_maker_list):
    """Return list of individual decision makers."""

    individual_dms = []
    for dm in decision_maker_list:
        individual_dms += dm.decision_makers
    
    return individual_dms


def get_human_and_model_decision_makers(decision_maker_list):

    individual_dms = get_individual_decision_makers(decision_maker_list)
    humans = []
    models = []
    for dm in individual_dms:
        if dm.startswith("subject-"):
            humans.append(dm)
        else:
            models.append(dm)
    return humans, models


def decision_maker_to_attributes(decision_maker_name, decision_maker_list):
    """Based on str 'decision_maker', return dict of attributes."""

    assert type(decision_maker_name) is str

    for dm in decision_maker_list:
        for individual_dm in dm.decision_makers:
            if decision_maker_name == "humans" and individual_dm.startswith("subject-") or \
                decision_maker_name.replace("-", "_") == individual_dm.replace("-", "_"):
                return {"color": dm.color,
                        "marker": dm.marker,
                        "plotting_name": dm.plotting_name,
                        "ID": dm.ID}
    raise ValueError("No attributes found for decision maker "+decision_maker_name)
