#!/bin/python
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import os
import argparse
import glob
import json

start_time = time.time()

"""Bounding box visualizer. Draws a bounding box on an image from an already labelled box. Imports box using json files.
 Use to verify the accuracy of labels en masse. Faster and smaller than transferring labelled pictures everywhere."""

# Options for changing default settings.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_path', help='Path to the image or folder of images to overlay.', required=False)
parser.add_argument('-d', '--dataset_path', help='Path to dataset. Output folder will be created here,', required=False)
parser.add_argument('-j', '--json_path', help='Path to the image or folder of images to overlay.', required=False)
parser.add_argument('-o', '--output_path', help='Absolute path for output folder. By default, appears in dataset dir.')
args = parser.parse_args()
image_path = args.image_path
dataset_path = args.dataset_path
json_path = args.json_path
output_path = args.output_path


def initialize():

    # Defaults. Change if argument received.
    if dataset_path:
        dataset_dir = dataset_path
    else:
        dataset_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + '/Dataset')

    if output_path:
        output_dir = output_path
    else:
        output_dir = dataset_dir + '/Bounding-Boxes'

    if json_path:
        json_dir = json_path
    else:
        json_dir = os.path.dirname(__file__) + '/instances_train2014.json'

    if image_path:
        image_dir = image_path
    else:
        image_dir = glob.glob(dataset_dir + '/**/*.png', recursive = True)
        image_dir = [os.path.abspath(image) for image in image_dir]

    os.makedirs(output_dir, exist_ok = True)
    print("Dataset: {}".format(dataset_dir))
    print("Output: {}".format(output_dir))
    print("Image folder: {}".format(image_dir))
    print("json file: {}".format(json_dir))
    return dataset_dir, output_dir, image_dir, json_dir


# Set up directories and lists.
dataset_dir, output_dir, image_dir, json_dir = initialize()
image_name = [os.path.basename(image) for image in image_dir]

for idx, image in enumerate(image_dir):

    # Load image and plot it.
    img = mpimg.imread(image)
    plt.imshow(img)
    axis = plt.gca()

    # Load json produced by buildWithLibrary and get the annotations.
    json_dir = os.path.abspath(json_dir)  # python2 __file__ is not absolute
    with open(json_dir) as json_data:
        data = json.load(json_data)
    annotation_set = data["annotations"]

    # Search annotations for the image and get corresponding bounding boxes.
    box_set = []
    for annotation in annotation_set:
        if annotation['image_id'] == image_name[idx]:
            box_set.append(annotation['bbox'])

    # Draw the bounding boxes for each box on the image.
    for box in box_set:
        axis.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'))

    # Save the figure.
    plt.savefig(output_dir + "/" + image_name[idx])
    print("Image {} saved.".format(image_name[idx]))
    plt.cla()

print("Runtime: %s seconds for %s images" % ((time.time() - start_time), len(image_dir)))
