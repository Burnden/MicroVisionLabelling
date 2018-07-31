import time
import os
import threading
from subprocess import Popen, PIPE
import re
import json
import matplotlib.pyplot as plt
import numpy as np

# Assume darknet folder is in user's home directory. If not, change it here.
"""# For big YOLO:
weights_file = "/home/amyxxxv/darknet/darknet53.conv.74"
config_file = "/home/amyxxxv/darknet/cfg/yolov3.cfg"
"""
# For tiny YOLO:
weights_file = "/home/amyxxxv/darknet/yolov3-tiny.conv.15"
config_file = "/home/amyxxxv/darknet/cfg/yolov3-tiny.cfg"

data_file = "/home/amyxxxv/darknet/cfg/crystals.data"
output_folder = "/home/amyxxxv/MicroVisionLabelling/yoloTrainingLogs"
yolo_command = "/home/amyxxxv/darknet/darknet detector train " + data_file + " " + config_file + " " + weights_file


def cleanup(log_folder):
    if os.path.exists(log_folder + '/yolo.log') and os.path.exists(log_folder + '/yololog.json'):
        os.makedirs(os.path.join(log_folder, "Saved Logs"), exist_ok=1)
        os.rename((log_folder + "/yolo.log"), (log_folder + "/Saved Logs" + '/yolo.log'))
        os.rename((log_folder + "/yololog.json"),(log_folder + "/Saved Logs" + '/yololog.json'))
        os.rename((log_folder + "/Loss-Visualization.png"), log_folder + "/Saved Logs" + "/Loss-Visualization.png")
        print("Files were moved to saved logs subfolder. Make sure to save and rename them to prevent overwriting"
              " on next run. Start in 3 sec.")
        time.sleep(3)


def plotter(xpt, ypt, need_to_set_up_a_plot, output_folder, x_list, y_list):
    if need_to_set_up_a_plot == 1:  # I.e. is this the first iteration since we started running?
        plt.xlabel('Iteration Number')
        plt.ylabel('Average Loss')
        plt.title('Average Loss vs. Iteration Number')

    x_list.append(xpt)
    y_list.append(ypt)

    if not len(x_list) % 10:
        plt.cla()
        x_array = np.array(x_list)
        y_array = np.array(y_list)
        curve = np.polyfit(x_array, y_array, 4)
        plt.plot(x_list, y_list, 'r.')
        plt.plot(x_array, np.polyval(curve, x_array), 'b-')

    else:
        plt.plot(xpt, ypt, 'r.')

    plt.savefig(output_folder + '/Loss-Visualization.png')


"""
def sanitizer(start, logfile):
    false_nums = re.findall("r:\s\d+", logfile)
    map(lstrip("r: "), false_nums)
    true_nums = range(start, int(false_nums[-1] - start))
    with open(logfile, 'w') as f:
        for idx, line in enumerate(logfile):
            line = line.split(false_nums[idx], maxsplit=1)
            line = str[line[0], true_nums[idx], line[1]]
            f.write(line)
"""


def logger(output_folder):
    first_iter = True
    os.makedirs(output_folder, exist_ok=True)
    cummu_x = []
    cummu_y = []
    while 1:
        buf = ""
        while 1:
            if re.search("images", buf):  # Wait until the iteration finishes, then log it.
                print(buf)
                break
            else:
                ch = yolo_process.stdout.readline()
                buf += ch.decode('utf-8')

        iter_num = re.findall("\d+:", buf)
        iter_num = (iter_num[0])[:-1]
        loss = re.findall("\d+.\d+ avg", buf)
        loss = (loss[0])[:-4]

        with open(output_folder + "/yolo.log", "a+") as logfile:
            write_me = "Iteration Number: " + iter_num + " - avg loss: " + loss + "\n"
            logfile.write(write_me)
        with open(output_folder + "/yololog.json", 'w') as logfile:
            write_me = {"Iteration_number": iter_num, "avg_loss": loss}
            json.dump(write_me, logfile)
            print("Log: {}".format(iter_num))
        plotter(int(iter_num), float(loss), first_iter, output_folder, cummu_x, cummu_y)
        first_iter = False
        """
        if not (iter_num % 100):  # Break up the numbers every 100 iterations. Can use for data analysis later.
            os.rename(output_folder + "/yolo.log", (output_folder + "/" + iter_num_string + "-iterations"))
            print("Logged {} iterations successfully, check the file.".format(iter_num))
        """


# Run Yolo on main thread, sub-threads used to log.
cleanup(output_folder)
yolo_process = Popen(yolo_command, stdout=PIPE, shell=True, cwd="/home/amyxxxv/darknet")
t = threading.Thread(target=logger, args=[output_folder])
t.start()
t.join()
