#!/usr/bin/env python3

"""
Create dataset and experiments.
A dataset is a directory with subdirectories, one subdir per class.
An experiment is a directory subdirectories, one subdir per participant.
"""

import os
from os.path import join as pjoin
from os import listdir as ld
import numpy as np
import shutil
import sys
from PIL import Image
import numpy as np
import math
from torchvision import transforms

from ..helper import human_categories as hc
from .. import constants as consts


def resize_crop_image(input_file,
                      resize_size,
                      crop_size):
    """Replace input_file with resized and cropped version (png)."""

    img = Image.open(input_file)
    t = transforms.Compose([transforms.Resize(resize_size),
                            transforms.CenterCrop(crop_size)])
    new_img = t(img)
    os.remove(input_file)
    new_img.save(input_file.replace(".JPEG", ".png"), 'png')


def create_dataset(original_dataset_path,
                   target_dataset_path,
                   rng,
                   min_num_imgs_per_class,
                   max_num_imgs_per_class,
                   target_resize_size,
                   target_crop_size):
    "Create a balanced dataset from a larger (potentially unbalanced) dataset."""

    categories = hc.HumanCategories()

    class_count_dict = dict()
    image_path_dict = dict()
    
    for human_category in sorted(hc.get_human_object_recognition_categories()):
        class_count_dict[human_category] = 0
        image_path_dict[human_category] = list()
 
    for c in sorted(os.listdir(original_dataset_path)):
        human_category = categories.get_human_category_from_WNID(c)
        if human_category is not None:
            class_count_dict[human_category] += len(os.listdir(pjoin(original_dataset_path,
                                                                     c)))
            for image_name in sorted(os.listdir(pjoin(original_dataset_path, c))):
                image_path_dict[human_category].append(pjoin(original_dataset_path,
                                                             c, image_name))

    count = 0
    maximum = 0
    minimum = np.Inf
    for c in sorted(os.listdir(original_dataset_path)):
        num = len(os.listdir(pjoin(original_dataset_path, c)))
        count += num
        if num > maximum:
            maximum = num
        if num < minimum:
            minimum = num

    min_16_classes = np.Inf
    for k, v in class_count_dict.items():
        if v < min_16_classes:
            min_16_classes = v

    print("Total image count: "+str(count))
    print("Max #images per class: "+str(maximum))
    print("Min #images per class: "+str(minimum))
    print("Min #images within 16 classes: "+str(min_16_classes))
    print(class_count_dict)

    assert min_16_classes >= min_num_imgs_per_class, "not enough images"
    num_imgs_per_target_class = max_num_imgs_per_class
    if min_16_classes < num_imgs_per_target_class:
        num_imgs_per_target_class = min_16_classes

    if not os.path.exists(target_dataset_path):
        print("Creating directory "+target_dataset_path)
        os.makedirs(target_dataset_path) 
    else:
        raise OSError("target dataset already exists: "+target_dataset_path)

    for human_category in sorted(hc.get_human_object_recognition_categories()):
        print("Creating category "+human_category)
        category_dir = pjoin(target_dataset_path, human_category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        num_images = class_count_dict[human_category]
        assert num_images >= min_16_classes, "not enough images found"

        choice = rng.choice(num_images, num_imgs_per_target_class, replace=False)

        assert len(choice) <= len(image_path_dict[human_category])
        assert len(choice) == num_imgs_per_target_class

        for image_index in choice:
            image_index_str = str(image_index+1)
            while len(image_index_str) < 4:
                image_index_str = "0"+image_index_str

            image_path = image_path_dict[human_category][image_index]
            target_image_path = pjoin(target_dataset_path, human_category,
                                      human_category+"-"+image_index_str+"-"+image_path.split("/")[-1].replace("_", "-"))
            shutil.copyfile(image_path, target_image_path)
            resize_crop_image(target_image_path, target_resize_size,
                              target_crop_size)
            

def create_experiment(expt_name,
                      expt_abbreviation,
                      expt_source_dir,
                      expt_target_dir,
                      only_dnn=True,
                      num_subjects=1,
                      rng=None):
    """Create human / CNN experiment.

    parameters:
    - only_dnn: boolean indicating whether this is a DNN experiment
              or not (if not, a human experiment will be created.)
    """

    if not only_dnn:
        assert rng is not None, "Please specify random number generator (rng)!"

    assert("_" not in expt_name), "no '_' in experiment name!"
    assert(os.path.exists(expt_source_dir)), "directory "+expt_source_dir+" does not exist."

    for i in range(0, num_subjects+1):

        if i==0:
            subject_abbreviation = "dnn"
            subject_name="dnn"
        else:
            subject_abbreviation = "s"+get_leading_zeros(i, 2)
            subject_name = "subject-"+get_leading_zeros(i, 2)
        print("Creating experiment for subject: '"+subject_name+"'")

        target_dir = pjoin(expt_target_dir, expt_name,
                           subject_name, "session-1")

        if os.path.exists(target_dir):
            print("Error: target directory "+target_dir+" does already exist.")
            sys.exit(1)
        else:
            os.makedirs(target_dir)

        img_list = []
        for c in sorted(hc.get_human_object_recognition_categories()):
            for x in sorted(ld(pjoin(expt_source_dir, c))):
                input_file = pjoin(expt_source_dir, c, x)
                img_list.append(input_file)
                
        order = np.arange(len(img_list))
        if i != 0:
            rng.shuffle(order)

        for i, img_index in enumerate(order):

            input_file = img_list[img_index]
            imgname = input_file.split("/")[-1]
            correct_category = input_file.split("/")[-2]
            condition = "0"
            target_image_path = pjoin(target_dir,
                                      (get_leading_zeros(i+1)+"_"+
                                       expt_abbreviation+"_"+
                                       subject_abbreviation+"_"+
                                       condition+"_"+
                                       correct_category+"_"+
                                       "00_"+
                                       imgname))

            shutil.copyfile(input_file, target_image_path)


def get_leading_zeros(num, length=4):
    return ("0"*length+str(num))[-length:]
