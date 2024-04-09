# from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from keras.models import Model
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.layers import ReLU
from keras.callbacks import TensorBoard
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras import losses
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

from train import model, index_array

# Step size for parameter iteration
step = 0.5

# Arrays to store the output data
out_s11 = np.empty((0, 10))
out_gain = np.empty((0, 10))
out_both = np.empty((0, 10))
out_max = np.empty((0, 10))
out_total = np.empty((0, 10))

# File paths for saving the output data
S11_Save_path = "Output_Bandwidth1000+.txt"
Gain_Save_path = "Output_Gain9.5+.txt"
Both_Save_path = "Output_Both.txt"
MAX_Save_path = "Output_Max.txt"
Total_Save_path = "Output_Total.txt"

# Calculate the total number of iterations based on the parameter ranges and step size
total_num = (5 / 1 + 1) * (12 / step + 1) * (8 / step + 1) * (4 / step + 1) * (5 / step + 1) * (4 / step + 1)
print("Total Number:", total_num)

# Initialize counters and variables
num = 0
delta = 1000  # Interval for progress updates
Gain_max = 0  # Maximum gain encountered
BandWidth_max = 0  # Maximum bandwidth encountered

# Iterate over the parameter space
for h in range(6 - 1, 10 + 1):  # Iterate over 'h' parameter
    for Scale_X_m10 in range(int((70 - 1) / step), int((80 + 1) / step) + 1):  # Iterate over 'Scale_X' parameter
        Scale_X = Scale_X_m10 * step
        for Scale_Y_m10 in range(int((38 - 1) / step), int((44 + 1) / step) + 1):  # Iterate over 'Scale_Y' parameter
            Scale_Y = Scale_Y_m10 * step
            for Offset_y_m10 in range(int((-7 - 1) / step), int((-5 + 1) / step) + 1):  # Iterate over 'Offset_y' parameter
                Offset_y = Offset_y_m10 * step
                for Scale_Slot_m10 in range(int((9 - 1) / step), int((12 + 1) / step) + 1):  # Iterate over 'Scale_Slot' parameter
                    Scale_Slot = Scale_Slot_m10 * step
                    for Uw3_m10 in range(int((10 - 1) / step), int((12 + 1) / step) + 1):  # Iterate over 'Uw3' parameter

                        Uw3 = Uw3_m10 * step

                        # Create an index array for the current parameters
                        index = np.array([h, Scale_X, Scale_Y, Offset_y, Scale_Slot, Uw3]).reshape(1, 6)
                        # Normalize the index array
                        index_norm = (index - index_array[0, :]) / (index_array[1, :] - index_array[0, :])
                        input = index_norm
                        # Predict the output using the trained model
                        pre_y = model.predict(input)

                        # Reverse the normalization for S11 and Gain
                        pre_y[:, 1:] = -70 * pre_y[:, 1:]
                        pre_y[:, 0] = 5 * pre_y[:, 0] + 5
                        S11 = pre_y[:, 1:]
                        Gain = pre_y[:, 0]

                        # Update the iteration counter
                        num += 1
                        # Print progress and current parameters
                        if num % delta == 0:
                            print("%d / %d is Done!" % (num, total_num))
                            print("Current Parameters:", index[0])
                            print()

                        # Calculate the bandwidth
                        result = np.empty((0, 9))
                        band = [np.where(line <= -10)[0] / 200 + 1.5 for line in S11]
                        for i in range(1):
                            a, b = [], []
                            if band[i].any():
                                a = [band[i][0]]
                                for j in range(1, len(band[i])):
                                    if round((band[i][j] - band[i][j - 1]) * 1000) / 1000 == 0.005:
                                        a.append(band[i][j])
                                    else:
                                        break
                                if j < len(band[i]) - 1:
                                    j0 = j
                                    b = [band[i][j0]]
                                    for j in range(j0 + 1, len(band[i])):
                                        if round((band[i][j] - band[i][j - 1]) * 1000) / 1000 == 0.005:
                                            b.append(band[i][j])
                                        else:
                                            break
                            if not (2.45 in a):
                                a = []
                            if not (2.45 in b):
                                b = []
                            # if not(a == []) and int((a[-1] - a[0]) * 1000) / h[i] >= 130:
                            if not (a == []):
                                # print('ID:', i + 1, ' Band:', a[0], 'GHz -', a[-1], 'GHz  Band Width:', int((a[-1] - a[0]) * 1000), 'MHz  H:', h[i], 'mm  B/H:', int(int((a[-1] - a[0]) * 1000) / h[i]))
                                result = np.append(result, [
                                    np.concatenate((index[i], np.array([a[0], a[-1], (a[-1] - a[0]) * 1000])),
                                                   axis=0)], axis=0)
                            # if not (b == []) and int((b[-1] - b[0]) * 1000) / h[i] >= 130:
                            elif not (b == []):
                                # print('ID:', i + 1, ' Band:', b[0], 'GHz -', b[-1], 'GHz  Band Width:', int((b[-1] - b[0]) * 1000), 'MHz  H:', h[i], 'mm  B/H:', int(int((b[-1] - b[0]) * 1000) / h[i]))
                                result = np.append(result, [
                                    np.concatenate((index[i], np.array([b[0], b[-1], (b[-1] - b[0]) * 1000])),
                                                   axis=0)], axis=0)
                            else:
                                result = np.append(result,
                                                   [np.concatenate((index[i], np.array([0, 0, 0])), axis=0)],
                                                   axis=0)

                        # Append the current result to the total output array
                        out_total = np.append(out_total, [
                            np.concatenate((index[0], np.array([result[0][-3], result[0][-2], result[0][-1], Gain[0]])),
                                           axis=0)], axis=0)

                        # Check if the bandwidth is greater than 1000 MHz
                        if result[0, -1] > 1000:
                            # Append the result to the S11 output array
                            out_s11 = np.append(out_s11, [np.concatenate(
                                (index[0], np.array([result[0][-3], result[0][-2], result[0][-1], Gain[0]])), axis=0)],
                                                axis=0)

                        # Check if the gain is greater than 9.5
                        if Gain[0] > 9.5:
                            # Append the result to the gain output array
                            out_gain = np.append(out_gain, [np.concatenate(
                                (index[0], np.array([result[0][-3], result[0][-2], result[0][-1], Gain[0]])), axis=0)],
                                                 axis=0)

                        # Check for specific conditions and append to the both output array
                        if h <= 7 and result[0, -1] > 700 and Gain[0] > 9.0:
                            out_both = np.append(out_both, [np.concatenate(
                                (index[0], np.array([result[0][-3], result[0][-2], result[0][-1], Gain[0]])), axis=0)],
                                                 axis=0)

                        # Update the maximum gain and bandwidth encountered
                            if Gain[0] > Gain_max:
                                Gain_max = Gain[0]
                                arg_gain = np.concatenate(
                                    (index[0], np.array([result[0][-3], result[0][-2], result[0][-1], Gain[0]])),
                                    axis=0)

                            if result[0, -1] > BandWidth_max:
                                BandWidth_max = result[0, -1]
                                arg_bandwidth = np.concatenate(
                                    (index[0], np.array([result[0][-3], result[0][-2], result[0][-1], Gain[0]])),
                                    axis=0)

# Save the output data to text files
np.savetxt(S11_Save_path, out_s11, fmt='%.3f')
np.savetxt(Gain_Save_path, out_gain, fmt='%.3f')
np.savetxt(Both_Save_path, out_both, fmt='%.3f')

# Append the maximum gain and bandwidth results to the out_max array
out_max = np.append(out_max, [arg_gain], axis=0)
out_max = np.append(out_max, [arg_bandwidth], axis=0)

# Save the out_max array
np.savetxt(MAX_Save_path, out_max, fmt='%.3f')

# Save the total output array
np.savetxt(Total_Save_path, out_total, fmt='%.3f')
