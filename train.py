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

# ----------| 读取 |---------- #
data = np.loadtxt(r'train_data.txt')

np.random.seed(0)
np.random.shuffle(data)

index = data[:, :6]  # 索引
gain = data[:, 6]  # 增益
S11 = data[:, 7:]  # 1.5-3.5 S11幅度
[row, column] = np.shape(S11)
S11 = -S11

# ----------| 预处理 |---------- #
S11[np.where(S11 < 0)] = 0
max_S11 = 70
min_S11 = 0
S11_norm = S11/max_S11

max_gain = 10
min_gain = 5
gain_norm = (gain-min_gain)/(max_gain-min_gain)

index1_h_min = 6-1
index1_h_max = 9+1
index2_Scale_X_min = 70-1
index2_Scale_X_max = 80+1
index3_Scale_Y_min = 38-1
index3_Scale_Y_max = 44+1
index4_Offset_y_min = -7-1
index4_Offset_y_max = -5+1
index5_Scale_Slot_m10_min = 9-1
index5_Scale_Slot_m10_max = 12+1
index6_Uw3_min = 10-1
index6_Uw3_max = 12+1
index_array = np.array([[index1_h_min, index2_Scale_X_min, index3_Scale_Y_min, index4_Offset_y_min, index5_Scale_Slot_m10_min, index6_Uw3_min], [index1_h_max, index2_Scale_X_max, index3_Scale_Y_max, index4_Offset_y_max, index5_Scale_Slot_m10_max, index6_Uw3_max]])
index_norm = (index-index_array[0, :])/(index_array[1, :]-index_array[0, :])

train_number = 3000
train_S11_norm = S11_norm[:train_number, :]
train_gain_norm = gain_norm[:train_number]
train_index_norm = index_norm[:train_number, :]

test_S11_norm = S11_norm[train_number:, :]
test_gain_norm = gain_norm[train_number:]
test_index_norm = index_norm[train_number:, :]

input_train = train_index_norm  # (3000, 6)

output1_train = train_gain_norm
output1_train = output1_train.reshape(train_number, 1)  # (3000, 1)
output2_train = train_S11_norm  # (3000, 401)
output_train = np.hstack((output1_train, output2_train))  # (3000, 402)

input_test = test_index_norm  # (3000, 6)

output1_test = test_gain_norm
output1_test = output1_test.reshape(3456 - train_number, 1)  # (456, 1)
output2_test = test_S11_norm  # (456, 401)
output_test = np.hstack((output1_test, output2_test))  # (456, 402)

# ----------| 训练 |---------- #
model = Sequential()
model.add(keras.layers.Dense(8, input_dim=input_train.shape[1], activation="relu"))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(402, activation="sigmoid"))
model.summary()

# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
filepath = "saved-model-{epoch:02d}.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath, monitor="val_loss", mode="min", save_weights_only=True, save_best_only=False, verbose=1, period=1000)

model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model_fit = model.fit(input_train, output_train, batch_size=512,
                      epochs=5000, verbose=0,
                      validation_split=0.2,
                      callbacks=[checkpoint])
                      # callbacks=[early_stopping])

# 可视化
plt.figure()
plt.plot(model_fit.history["loss"])
plt.plot(model_fit.history["val_loss"])
plt.xlabel("iters")
plt.ylabel("loss")
plt.show()

# 读取模型
model.load_weights("saved-model-5000.h5")

# 对训练集上进行预测
pre_y = model.predict(input_train)

# print("mean absolute error:", keras.metrics.mean_absolute_error(output_train, pre_y))
# print("mean squared error:", keras.metrics.mean_squared_error(output_train, pre_y))
#
# output_train[:, 1:] = -70 * output_train[:, 1:]
# output_train[:, 0] = 5 * output_train[:, 0] + 5
# pre_y[:, 1:] = -70 * pre_y[:, 1:]
# pre_y[:, 0] = 5 * pre_y[:, 0] + 5
#
# # Plot S11
# for i in range(3):
#     index = np.random.randint(0, 2999)
#     plt.plot(np.arange(401), output_train[index, 1:])
#     plt.plot(np.arange(401), pre_y[index, 1:])
#     plt.title("Train%d, Index%d" % (i + 1, index))
#     plt.show()
#
# # Plot Gain
# plt.scatter(np.arange(3000), output_train[:, 0], marker='o', s=10, c=None, edgecolors='r')
# plt.scatter(np.arange(3000), pre_y[:, 0], marker='x', s=10, c='black')
# plt.title("Training Gain")
# plt.show()

# 对测试集上进行预测
pre_y = model.predict(input_test)
print("mean absolute error:", keras.metrics.mean_absolute_error(output_test, pre_y))
print("mean squared error:", keras.metrics.mean_squared_error(output_test, pre_y))

output_test[:, 1:] = -70 * output_test[:, 1:]
output_test[:, 0] = 5 * output_test[:, 0] + 5
pre_y[:, 1:] = -70 * pre_y[:, 1:]
pre_y[:, 0] = 5 * pre_y[:, 0] + 5
print(input_test)

# Plot S11
# for i in range(10):
#     index = np.random.randint(0, 455)
#     plt.plot(np.arange(401), output_test[index, 1:], 'black')
#     plt.plot(np.arange(401), pre_y[index, 1:])
#     plt.title("Test%d, Index%d" % (i + 1, index))
#     plt.show()




# Plot Gain
plt.scatter(np.arange(100), output_test[:100, 0], marker='o', s=40, c='White', edgecolors='r')
plt.scatter(np.arange(100), pre_y[:100, 0], marker='x', s=40, c='black')
plt.title("Testing Gain")
plt.show()
