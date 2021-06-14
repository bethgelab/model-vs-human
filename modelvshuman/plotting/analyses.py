"""
Analyses based on .csv files containing experimental data.
"""

import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from copy import deepcopy

from .. import constants as c
from  ..helper import human_categories as hc
from . import decision_makers as dm


class Analysis(ABC):
    figsize = (7, 6)

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _check_dataframe(df):
        assert len(df) > 0, "empty dataframe"

    @abstractmethod
    def analysis(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_result_df(self, *args, **kwars):
        pass

    @property
    @abstractmethod
    def num_input_models(self):
        """Return number of input data frames for analysis.

        E.g. if analysis compares two observers/models, this
        number will be 2.
        """
        pass


class ConfusionAnalysis(Analysis):

    num_input_models = 1

    def __init__(self):
        super().__init__()
        self.plotting_name = "confusion-matrix"


    def analysis(self, df,
                 categories=hc.get_human_object_recognition_categories(),
                 include_NA=True):

        df = deepcopy(df)

        c = categories
        if include_NA:
            c = ["na"]+c
        df['object_response'] = pd.Categorical(df.object_response,
                                               categories=c)

        confusion_matrix = pd.crosstab(df['object_response'], df['category'],
                                       dropna=False)

        confusion_matrix = 100.0*confusion_matrix.astype('float') / confusion_matrix.sum(axis=0)[np.newaxis, :]
        return confusion_matrix


    def get_result_df():
        pass



class ShapeBias(Analysis):
    """Reference: Geirhos et al. ICLR 2019
    https://openreview.net/pdf?id=Bygh9j09KX
    """
    num_input_models = 1

    def __init__(self):
        super().__init__()
        self.plotting_name = "shape-bias"


    def analysis(self, df):

        self._check_dataframe(df)

        df = df.copy()
        df["correct_texture"] = df["imagename"].apply(self.get_texture_category)
        df["correct_shape"] = df["category"]

        # remove those rows where shape = texture, i.e. no cue conflict present
        df2 = df.loc[df.correct_shape != df.correct_texture]
        fraction_correct_shape = len(df2.loc[df2.object_response == df2.correct_shape]) / len(df)
        fraction_correct_texture = len(df2.loc[df2.object_response == df2.correct_texture]) / len(df)
        shape_bias = fraction_correct_shape / (fraction_correct_shape + fraction_correct_texture)

        result_dict = {"fraction-correct-shape": fraction_correct_shape,
                       "fraction-correct-texture": fraction_correct_texture,
                       "shape-bias": shape_bias}
        return result_dict


    def get_result_df(self):
        pass


    def get_texture_category(self, imagename):
        """Return texture category from imagename.

        e.g. 'XXX_dog10-bird2.png' -> 'bird
        '"""
        assert type(imagename) is str

        # remove unneccessary words
        a = imagename.split("_")[-1]
        # remove .png etc.
        b = a.split(".")[0]
        # get texture category (last word)
        c = b.split("-")[-1]
        # remove number, e.g. 'bird2' -> 'bird'
        d = ''.join([i for i in c if not i.isdigit()])
        return d


class ErrorConsistency(Analysis):
    """Reference: Geirhos, Meding & Wichmann, NeurIPS 2020
    https://arxiv.org/abs/2006.16736
    """

    num_input_models = 2

    def __init__(self):
        super().__init__()
        self.plotting_name = "error-consistency"
        self.ylabel = "Error consistency (kappa)"
        self.ylim = (-0.2, 1)
        self.height_line_for_chance = 0.0
        self.figsize = (8.8, 7.14)


    def error_consistency(self, expected_consistency, observed_consistency):
        """Return error consistency as measured by Cohen's kappa."""

        assert expected_consistency >= 0.0
        assert expected_consistency <= 1.0
        assert observed_consistency >= 0.0
        assert observed_consistency <= 1.0

        if observed_consistency == 1.0:
            return 1.0
        else:
            return (observed_consistency - expected_consistency) / (1.0 - expected_consistency)


    def analysis(self, df1, df2):
        """Return error consistency"""

        self._check_dataframe(df1)
        self._check_dataframe(df2)
        assert len(df1) == len(df2)

        num_trials = len(df1)
        p1 = SixteenClassAccuracy().analysis(df1)["16-class-accuracy"]
        p2 = SixteenClassAccuracy().analysis(df2)["16-class-accuracy"]
        expected_consistency = p1 * p2 + (1 - p1) * (1 - p2)

        # sort such that all trials are in the same order
        df1 = df1.sort_values(by="image_id")
        df2 = df2.sort_values(by="image_id")
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        assert df1["image_id"].equals(df2["image_id"])

        df1 = df1.copy()
        df2 = df2.copy()
        df1["is_correct"] = df1.object_response == df1.category
        df2["is_correct"] = df2.object_response == df2.category
        observed_consistency = (df1.is_correct == df2.is_correct).sum() / len(df1)

        error_consistency = self.error_consistency(expected_consistency=expected_consistency,
                                                   observed_consistency=observed_consistency)

        return {"expected-consistency": expected_consistency,
                "observed-consistency": observed_consistency,
                "error-consistency":    error_consistency}


    def get_result_df(self, df, decision_makers, experiment, column="error-consistency"):
        """Return mean consistency between decision_makers and human observers."""

        result_df = pd.DataFrame(columns=['subj', 'condition',
                                          'yvalue', 'decision-maker-ID',
                                          'colour'])


        humans, models = dm.get_human_and_model_decision_makers(decision_makers)
        assert len(humans) > 0, "Error consistency between humans and models can "+ \
                                "only be computed if human observers are included " + \
                                "as decision makers."

        def get_result(result_df, df, dm1, condition):
            values = []
            for h in humans:
                if dm1 != h:
                    df1 = df.loc[(df["condition"]==condition) & (df["subj"]==dm1)]
                    df2 = df.loc[(df["condition"]==condition) & (df["subj"]==h)]
                    r = self.analysis(df1, df2)
                    values.append(r["error-consistency"])
            return np.mean(values)


        for c in experiment.data_conditions:
            for m in models:
                yvalue = get_result(result_df, df, m, condition=c)
                attr = dm.decision_maker_to_attributes(m, decision_makers)
                result_df = result_df.append({'subj': attr["plotting_name"],
                                              'condition': c,
                                              'yvalue': yvalue,
                                              'decision-maker-ID': attr["ID"]},
                                              ignore_index=True)
 
            hvalues = []
            for h in humans:
                hvalues.append(get_result(result_df, df, h, condition=c))

            attr = dm.decision_maker_to_attributes(h, decision_makers)
            result_df = result_df.append({'subj': attr["plotting_name"],
                                          'condition': c,
                                              'yvalue': np.mean(hvalues),
                                              'decision-maker-ID': attr["ID"]},
                                              ignore_index=True)

        return result_df


class XYAnalysis(Analysis):

    def get_result_df(self, df, decision_makers,
                      experiment):

        result_df = pd.DataFrame(columns=['subj', 'condition',
                                          'yvalue', 'decision-maker-ID',
                                          'colour'])

        for d in decision_makers:
            for c in experiment.data_conditions:
                subdat = df.loc[(df["condition"]==c) & (df["subj"].isin(d.decision_makers))]
                r = self.analysis(subdat)
                assert len(r) == 1, "Analysis unexpectedly returned more than one scalar."
                result_df = result_df.append({'subj': d.plotting_name,
                                              'condition': c,
                                              'yvalue': list(r.values())[0],
                                              'decision-maker-ID': d.ID},
                                              ignore_index=True)

        return result_df


class SixteenClassAccuracy(XYAnalysis):
    """Reference: Geirhos et al. NeurIPS 2018
    https://arxiv.org/abs/1808.08750
    """

    NUM_CATEGORIES = 16
    num_input_models = 1

    def __init__(self):
        super().__init__()
        self.ylabel = "Classification accuracy"
        self.plotting_name = "OOD accuracy"
        self.ylim = (0, 1)
        self.height_line_for_chance = 1 / self.NUM_CATEGORIES


    def analysis(self, df):
        """Return accuracy of responses."""

        self._check_dataframe(df)
        accuracy = len(df.loc[df.object_response == df.category]) / len(df)
        result_dict = {"16-class-accuracy": accuracy}
        return result_dict


class SixteenClassAccuracyDifference(XYAnalysis):
    """Difference between two accuracies (e.g. model and human)."""

    NUM_CATEGORIES = 16
    num_input_models = 2

    def __init__(self):
        super().__init__()
        self.ylabel = "Classification accuracy difference"
        self.plotting_name = "accuracy-difference"
        self.ylim = (0, 1)


    def analysis(self, df1, df2, norm=np.square):
        """Return accuracy of responses."""

        self._check_dataframe(df1)
        self._check_dataframe(df2)
        accuracy_1 = len(df1.loc[df1.object_response == df1.category]) / len(df1)
        accuracy_2 = len(df2.loc[df2.object_response == df2.category]) / len(df2)
        acc_difference = norm(accuracy_1 - accuracy_2)
        result_dict = {"16-class-accuracy-difference": acc_difference}
        return result_dict


class Entropy(XYAnalysis):
    """Reference: Geirhos et al. NeurIPS 2018
    https://arxiv.org/abs/1808.08750
    """

    num_input_models = 1

    def __init__(self, num_categories=16):
        super().__init__()
        self.ylabel = "Response distr. entropy [bits]"
        self.plotting_name = "entropy"
        self.ylim = (0, np.log2(num_categories))
        self.height_line_for_chance = np.log2(num_categories)


    def analysis(self, df):
        """Return response distribution entropy."""

        self._check_dataframe(df)

        entropy_per_subj = []
        for subj in df["subj"].unique():

            response_probabilities = get_percent_answers_per_category(df.loc[df["subj"]==subj])

            entropy = 0.0
            for p in response_probabilities:
                if abs(p) > 1e-5:  # ignore values too close to zero
                    entropy += p * np.log2(p)
            entropy = 0 - entropy
            entropy_per_subj.append(entropy)

        result_dict = {"entropy": np.mean(entropy_per_subj)}
        return result_dict


def get_analysis_list(df, conditions, analysis):
    """Apply analysis to all conditions within dataframe."""

    analysis_list = []
    for c in conditions:
        subset = df.loc[df.condition == c]
        analysis_result = analysis(subset)
        keys = list(analysis_result.keys())
        assert len(keys) == 1
        analysis_list.append(analysis_result[keys[0]])

    return analysis_list


def get_percent_answers_per_category(df):
    """Return a list of percentages, one per category.

    Helper function for entropy computation.
    Each value indicates the percentage of object_response
    for this category. The sum, thus, should be 1.0 (CNN)
    or close to 1 (humans, 'na' not included)
    """

    result = []
    num_answers = len(df)
    for category in df.category.unique():
        if str(category) != "na":
            result.append(len(df.loc[df.object_response == category]) / num_answers)

    return result
