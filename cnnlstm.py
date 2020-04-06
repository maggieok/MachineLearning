from random import random
from random import randint
from numpy import array
from numpy import zeros

from keras.layers import Input, Lambda, Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten, Dropout
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

reshape_size = 128
TIME_STEP = 3
data_root = 'E:/Thank/00/'
test_data_root = 'E:/Thank/02/'
seq_len = 3
out_len = 1
data_container =np.empty((TIME_STEP, 1, reshape_size, reshape_size), dtype="float32")
label_container = np.empty((1,), dtype="uint8")

test_data_container =np.empty((TIME_STEP, 1, reshape_size, reshape_size), dtype="float32")
test_label_container = np.empty((1,), dtype="uint8")

def update_sampled_img():
    all_lens = 0
    train_start = 807  # 807
    train_end = 900 # 1071
    data = []
    label = []
    for i in range(train_start, train_end):
        filePath = data_root + str(i)
        totalLens = len([x for x in os.listdir(filePath)])
        all_len = totalLens - (seq_len + out_len) + 1
        all_lens += all_len
    print("all_lens",all_lens)
    for i in range(train_start, train_end):
        filePath = data_root + str(i)
        folder = data_root + str(i)
        totalLens = len([x for x in os.listdir(filePath)])
        lens = totalLens - (seq_len + out_len) + 1
        datasum = np.zeros((lens, TIME_STEP, 1, reshape_size, reshape_size), dtype=np.float32)
        # 可预测的长度
        for j in range(lens):
            # 每四张图片作为输入长度
            for s in range(seq_len):
                # img = Image.open(filePath + f'/{i}_{j + s}.jpg')
                imgs = []
                # imgs值为此文件夹所有图片名
                for f in sorted(os.listdir(folder)):  # os.listdir(folder) :'764_0_2.jpg'
                    if f.endswith('jpg'):
                        imgs.append(os.path.join(folder, f))
                # 获取四张图片，放入data_container
                for img_a in imgs:
                    img_nums = f'{i}_{j + s}'
                    img_level_order_a = img_a.split("\\")[1].split(".")[0]
                    cycle_num_a = img_level_order_a.split("_")[0]
                    cycle_order_a = img_level_order_a.split("_")[1]
                    cycle_a = cycle_num_a + "_" + cycle_order_a
                    result_a = (cycle_a == img_nums)
                    if result_a:
                        img = Image.open(img_a)
                        img = img.resize((reshape_size, reshape_size))
                        img = np.asarray(img, dtype="float32")
                        data_container[s,:,:,:] = img / 255.0  #

            imgs1 = []
            # 获取第五章图片的台风等级，放入label_container
            img_nums_label = f'{i}_{j + s + 1}'
            for f in sorted(os.listdir(folder)):  # os.listdir(folder) :'764_0_2.jpg'
                if f.endswith('jpg'):
                    imgs1.append(os.path.join(folder, f))
            for img_label in imgs1:
                img_level_order = img_label.split("\\")[1].split(".")[0]
                cycle_num = img_level_order.split("_")[0]
                cycle_order = img_level_order.split("_")[1]
                cycle = cycle_num + "_" + cycle_order
                result = (cycle == img_nums_label)
                if result:
                    img_level = img_label.split("_")[-1].split(".")[0]
                    label.append(img_level)
            datasum[j, :, :, :, :] = data_container
        if data == []:
            data = datasum
        else:
            data = np.concatenate([data, datasum])

    data = np.array(data).reshape(-1, TIME_STEP,reshape_size, reshape_size, 1)
    data_len = len(data)
    print("data_len",data_len)
    label = np.array(label).reshape(-1, 1)

    lb = LabelBinarizer()
    label = lb.fit_transform(label)
    # print("label======", label)
    label_len = len(label)
    print("label_len",label_len)
    return data,label

def update_test_sampled_img():
    test_all_lens = 0
    test_start = 758  # 758
    test_end = 795 # 795
    test_data = []
    test_label = []
    for i in range(test_start, test_end):
        testFilePath = test_data_root + str(i)
        test_totalLens = len([x for x in os.listdir(testFilePath)])
        test_all_len = test_totalLens - (seq_len + out_len) + 1
        test_all_lens += test_all_len
    print("test_all_lens",test_all_lens)
    for i in range(test_start, test_end):
        testFilePath = test_data_root + str(i)
        test_folder = test_data_root + str(i)
        test_totalLens = len([x for x in os.listdir(testFilePath)])
        test_lens = test_totalLens - (seq_len + out_len) + 1
        test_datasum = np.zeros((test_lens, TIME_STEP, 1, reshape_size, reshape_size), dtype=np.float32)
        # 可预测的长度
        for j in range(test_lens):
            # 每四张图片作为输入长度
            for s in range(seq_len):
                # img = Image.open(filePath + f'/{i}_{j + s}.jpg')
                test_imgs = []
                # imgs值为此文件夹所有图片名
                for f in sorted(os.listdir(test_folder)):  # os.listdir(folder) :'764_0_2.jpg'
                    if f.endswith('jpg'):
                        test_imgs.append(os.path.join(test_folder, f))
                # 获取四张图片，放入data_container
                for test_img_a in test_imgs:
                    test_img_nums = f'{i}_{j + s}'
                    test_img_level_order_a = test_img_a.split("\\")[1].split(".")[0]
                    test_cycle_num_a = test_img_level_order_a.split("_")[0]
                    test_cycle_order_a = test_img_level_order_a.split("_")[1]
                    test_cycle_a = test_cycle_num_a + "_" + test_cycle_order_a
                    test_result_a = (test_cycle_a == test_img_nums)
                    if test_result_a:
                        test_img = Image.open(test_img_a)
                        test_img = test_img.resize((reshape_size, reshape_size))
                        test_img = np.asarray(test_img, dtype="float32")
                        test_data_container[s,:,:,:] = test_img / 255.0  #

            test_imgs1 = []
            # 获取第五章图片的台风等级，放入label_container
            img_nums_label = f'{i}_{j + s + 1}'
            for f in sorted(os.listdir(test_folder)):  # os.listdir(folder) :'764_0_2.jpg'
                if f.endswith('jpg'):
                    test_imgs1.append(os.path.join(test_folder, f))
            for test_img_label in test_imgs1:
                test_img_level_order = test_img_label.split("\\")[1].split(".")[0]
                test_cycle_num = test_img_level_order.split("_")[0]
                test_cycle_order = test_img_level_order.split("_")[1]
                cycle = test_cycle_num + "_" + test_cycle_order
                result = (cycle == img_nums_label)
                if result:
                    test_img_level = test_img_label.split("_")[-1].split(".")[0]
                    test_label.append(test_img_level)
            test_datasum[j, :, :, :, :] = test_data_container
        if test_data == []:
            test_data = test_datasum
        else:
            test_data = np.concatenate([test_data, test_datasum])

    test_data = np.array(test_data).reshape(-1, TIME_STEP,reshape_size, reshape_size, 1)
    test_data_len = len(test_data)
    print("test_data_len",test_data_len)
    test_label = np.array(test_label).reshape(-1, 1)
    test_lb = LabelBinarizer()
    test_label = test_lb.fit_transform(test_label)
    test_label_len = len(test_label)
    print("test_label_len",test_label_len)
    return test_data,test_label

# input  (batchsize,timesteps,width,height,channels)
# define the model
model = Sequential()
model.add(TimeDistributed(Conv2D(4, (2,2), activation= 'relu'), input_shape=(None,reshape_size,reshape_size,1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

# fit model
X, y = update_sampled_img()
model.fit(X, y, batch_size=128, epochs=100)

# evaluate model 新的随机序列上估计模型的学习能力
test_X, test_y = update_test_sampled_img()
loss, acc = model.evaluate(test_X, test_y, verbose=0)
print('loss: %f, acc: %f' % (loss, acc*100))

# # prediction on new data
# X, y = update_sampled_img()
# yhat = model.predict_classes(X, verbose=0)
# expected = "Right" if y[0]==1 else "Left"
# predicted = "Right" if yhat[0]==1 else "Left"
# print('Expected: %s, Predicted: %s' % (expected, predicted))









