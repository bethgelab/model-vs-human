"""
Generic evaluation functionality: evaluate on several datasets.
"""

import csv
import os
import shutil
import numpy as np
from math import isclose
from os.path import join as pjoin

from .. import constants as c

IMAGENET_LABEL_FILE = pjoin(c.CODE_DIR, "evaluation", "imagenet_labels.txt")

def print_performance_to_csv(model_name, dataset_name,
                             performance, metric_name,
                             data_parent_dir=c.PERFORMANCES_DIR):
    if not os.path.exists(data_parent_dir):
        os.makedirs(data_parent_dir)
    csv_file_path = pjoin(data_parent_dir, model_name + ".csv")
    newrow = [model_name, dataset_name,
              metric_name, performance]

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["subj", "dataset_name",
                             "metric_name", "performance"])
            writer.writerow(newrow)

    else:
        # check whether existing row needs to be overwritten, otherwise append
        # new row at the end.
        temp_file_path = csv_file_path.replace(".csv", "_temp.csv")
        with open(csv_file_path, 'r') as f:
            with open(temp_file_path, 'w') as t:
                reader = csv.reader(f)
                writer = csv.writer(t)

                has_overwritten_existing_row = False
                for i, row in enumerate(reader):
                    tmprow = row
                    if i >= 1:
                        assert row[0] == model_name
                        if row[1] == dataset_name and row[2] == metric_name:
                            tmprow = newrow
                            has_overwritten_existing_row = True
                    writer.writerow(tmprow)
                if not has_overwritten_existing_row:
                    writer.writerow(newrow)
        shutil.move(temp_file_path, csv_file_path)


def print_predictions_to_console(softmax_output, top_n=5,
                                 labels_path=IMAGENET_LABEL_FILE):
    """For each vector in the output batch: print predictions.

    For every vector of shape [1, 1000] in the output batch,
    a softmax is applied to the values. Then, the top_n
    (e.g. top 5) values are printed to the console.

    This can be used to check predictions for individual images.
    """

    assert type(softmax_output) is np.ndarray
    assert len(softmax_output.shape) == 2, \
        "len(softmax_output) needs to be 2 instead of " + str(len(softmax_output.shape))

    labels_file = open(labels_path)
    labels = labels_file.readlines()

    for z in range(softmax_output.shape[0]):
        print()
        print("Predictions for image no. " + str(z + 1))

        softmax_array = softmax_output[z, :]

        assert isclose(sum(softmax_array), 1.0, abs_tol=1e-5), \
            "Sum of softmax values equals " + str(sum(softmax_array)) + " instead of 1.0"

        argmax = softmax_array.argsort()[-top_n:][::-1]

        for i, argmax_value in enumerate(argmax):
            predicted_class = labels[argmax_value]
            predicted_class_index = predicted_class.split(":")[0]
            predicted_class_description = predicted_class.split(":")[1].replace("\n", "")

            print("({0}) {1:6.3f} % {2} [{3}]".format(i + 1,
                                                      100 * softmax_array[argmax_value],
                                                      predicted_class_description,
                                                      predicted_class_index))
        print()


class ResultPrinter():

    def __init__(self, model_name, dataset,
                 data_parent_dir=c.RAW_DATA_DIR):

        self.model_name = model_name
        self.dataset = dataset
        self.data_dir = pjoin(data_parent_dir, dataset.name)
        self.decision_mapping = self.dataset.decision_mapping
        self.info_mapping = self.dataset.info_mapping
        self.session_list = []

    def create_session_csv(self, session):

        self.csv_file_path = pjoin(self.data_dir,
                                   self.dataset.name + "_" +
                                   self.model_name.replace("_", "-") + "_" +
                                   session + ".csv")

        if os.path.exists(self.csv_file_path):
            # print("Warning: the following file will be overwritten: "+self.csv_file_path)
            os.remove(self.csv_file_path)

        directory = os.path.dirname(self.csv_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.index = 0

        # write csv file header row
        with open(self.csv_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["subj", "session", "trial",
                             "rt", "object_response", "category",
                             "condition", "imagename"])


    def print_batch_to_csv(self, object_response,
                           batch_targets, paths):

        for response, target, path in zip(object_response, batch_targets, paths):

            session_name, img_name, condition, category = self.info_mapping(path)
            session_num = int(session_name.split("-")[-1])

            if not session_num in self.session_list:
                self.session_list.append(session_num)
                self.create_session_csv(session_name)

            with open(self.csv_file_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.model_name,
                                 str(session_num), str(self.index+1),
                                 "NaN", response[0], category,
                                 condition, img_name])
            self.index += 1
