import numpy as np
import matplotlib.pyplot as plt
import json
import os


#  Make a text file with the names of all the images that are going to be tested.
def make_image_names_file(folder):
    with open(folder + "image_list.txt", 'w') as openlist:
        for image in folder:
            openlist.write("{}\n".format(image))


# Create list of yolo-predicted diagonal bounding box lengths as read from json. I.e. "Experimental" set.
def predicted_info(predicted_json_input):
    with open(predicted_json_input) as json_data:
        data = json.load(json_data)
        predicted_diag = data["length of diagonal"]
        predicted_certainty = data["certainty"]
    return predicted_diag, predicted_certainty


# Load json produced by buildWithLibrary and get the pre-labelled annotations. I.e. "Theoretical" set.
# Outputs percent error/
# Make sure the two input lists correspond to the same box and image.
def compare_diag_length(pred_diag, training_bbox_json):
    diag_list_train = []
    with open(training_bbox_json) as json_data:
        data = json.load(json_data)
        box_set = data["annotations"]['bbox']  # Search annotations for the bounding boxes.
        for idx, box in enumerate(box_set):
            diag_list_train[idx] = (box[2] ^ 2 + box[3] ^ 2) ** (1/2)

    percent_errors = (np.array(diag_list_train) - np.array(pred_diag)) / np.array(diag_list_train)
    percent_errors.tolist()
    return percent_errors


# Draw and save the histogram to an output folder. list1 is the data set to be plotted.
def draw_N_save(list1, out_folder):  # Make sure to include trailing /.
    list1.sort
    plt.hist(np.array(list1))
    plt.savefig(out_folder + list1)
    plt.close  # Prevent memory leak.
