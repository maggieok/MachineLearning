import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re

# Hyper Parameters
EPOCH = 20        # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 700
TIME_STEP = 4          # rnn time step / image height
INPUT_SIZE = 1024         # rnn input size / image width
LR = 0.001               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data


class MyDadaSet(Dataset):
    def __init__(self, seq_len=TIME_STEP, out_len=1):
        self.dataset_len = 1
        self.data_root = 'E:/0402/shuju3/'
        self.seq_len = seq_len
        self.out_len = out_len
        self.data_container = np.zeros((INPUT_SIZE, TIME_STEP), dtype=np.float32)
        # np.zeros()返回来一个给定形状和类型的用0填充的数组,默认numpy.float64
        self.label_container = list()
        self.data = None
        self.label = None

    def __getitem__(self, index):
        dt = self.data[index, ...]  # 相当于resize(),用来转换size大小;多行拼接成一行
        # print("dt", dt.shape) # [4, 1024]
        # print("self.label",self.label) # [0, 0, 3, 3, 3, 2, 2, 1, 1, 0],
        lb = self.label[index, ...]
        return dt, lb

    def __len__(self):
        return self.dataset_len

    def update_sampled_img(self):
        all_lens = 0
        train_start = 758
        train_end = 780
        iii = 0
        for i in range(train_start, train_end):
            filePath = self.data_root + str(i)
            totalLens = len([x for x in os.listdir(filePath)])
            all_len = totalLens - (self.seq_len + self.out_len) + 1
            all_lens += all_len
        datasum = np.zeros((INPUT_SIZE, TIME_STEP, all_lens), dtype=np.float32)
        labelsum = np.zeros((all_lens), dtype=np.int)
        ii = 0
        for i in range(train_start, train_end):
            filePath = self.data_root + str(i)
            folder = self.data_root + str(i)
            totalLens = len([x for x in os.listdir(filePath)])
            lens = totalLens - (self.seq_len + self.out_len) + 1
            # 可预测的长度
            for j in range(lens):
                # 每四张图片作为输入长度
                for s in range(self.seq_len):
                    # img = Image.open(filePath + f'/{i}_{j + s}.jpg')
                    imgs = []
                    # imgs值为此文件夹所有图片名
                    for f in sorted(os.listdir(folder)): # os.listdir(folder) :'764_0_2.jpg'
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
                            img = img.resize((32, 32))
                            img = np.array(img)
                            img = img.reshape(INPUT_SIZE)
                            self.data_container[..., s] = img / 255  # (32, 32, seq_len)

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
                        self.label_container.append(img_level)

                data1 = torch.from_numpy(self.data_container)
            datasum[...,ii] = data1
            ii += 1
        self.data = torch.from_numpy(datasum)
        self.data = self.data.permute(2, 1, 0) # [batch_size, 4, 1024]
        self.label_container = [int(x) for x in self.label_container]

        labelsum = np.array(self.label_container)
        labelsum = torch.from_numpy(labelsum)
        self.label = labelsum
        print("self.label.shape", self.label.shape)
        self.dataset_len = self.data.shape[0]
        print("self.dataset_len", self.dataset_len)
    print("over")



class vMyDadaSet(Dataset):
    def __init__(self, seq_len=TIME_STEP, out_len=1):
        self.vdataset_len = 1
        self.data_root = 'E:/0402/shuju3/'
        self.seq_len = seq_len
        self.out_len = out_len
        self.vdata_container = np.zeros((INPUT_SIZE, TIME_STEP), dtype=np.float32)
        # np.zeros()返回来一个给定形状和类型的用0填充的数组,默认numpy.float64
        self.vlabel_container = list()
        self.vdata = None
        self.vlabel = None

    def __getitem__(self, index):
        vdt = self.vdata[index, ...]  # 相当于resize(),用来转换size大小;多行拼接成一行
        # print("vvdt", vdt.shape) # [4, 1024]
        vlb = self.vlabel[index, ...]
        return vdt, vlb

    def __len__(self):
        return self.vdataset_len

    def vupdate_sampled_img(self):
        all_vlens = 0
        test_start = 807
        test_end = 813
        for i in range(test_start, test_end):
            filePath = self.data_root + str(i)
            totalLens = len([x for x in os.listdir(filePath)])
            all_vlen = totalLens - (self.seq_len + self.out_len) + 1
            all_vlens += all_vlen
        vdatasum = np.zeros((INPUT_SIZE, TIME_STEP,all_vlens), dtype=np.float32)
        vlabelsum = np.zeros((all_vlens), dtype=np.int)
        ii = 0
        for i in range(test_start, test_end):
            filePath = self.data_root + str(i)
            folder = self.data_root + str(i)
            totalLens = len([x for x in os.listdir(filePath)])
            vlens = totalLens - (self.seq_len + self.out_len) + 1
            # 可预测的长度
            for j in range(vlens):
                # 每四张图片作为输入长度
                for s in range(self.seq_len):
                    imgs = []
                    # imgs值为此文件夹所有图片名
                    for f in sorted(os.listdir(folder)): # os.listdir(folder) :'764_0_2.jpg'
                        if f.endswith('jpg'):
                            imgs.append(os.path.join(folder, f))
                    # 获取四张图片，放入data_container
                    for vimg_a in imgs:
                        vimg_nums = f'{i}_{j + s}'
                        vimg_level_order_a = vimg_a.split("\\")[1].split(".")[0]
                        vcycle_num_a = vimg_level_order_a.split("_")[0]
                        vcycle_order_a = vimg_level_order_a.split("_")[1]
                        vcycle_a = vcycle_num_a + "_" + vcycle_order_a
                        vresult_a = (vcycle_a == vimg_nums)
                        if vresult_a:
                            img = Image.open(vimg_a)
                            img = img.resize((32, 32))
                            img = np.array(img)
                            img = img.reshape(INPUT_SIZE)
                            self.vdata_container[..., s] = img / 255  # (32, 32, seq_len)

                vimgs1 = []
                # 获取第五张图片的台风等级，放入label_container
                img_nums_label = f'{i}_{j + s + 1}'
                for f in sorted(os.listdir(folder)):  # os.listdir(folder) :'764_0_2.jpg'
                    if f.endswith('jpg'):
                        vimgs1.append(os.path.join(folder, f))
                for vimg in vimgs1:
                    vimg_level_order = vimg.split("\\")[1].split(".")[0]
                    vcycle_num = vimg_level_order.split("_")[0]
                    vcycle_order = vimg_level_order.split("_")[1]
                    vcycle = vcycle_num + "_" + vcycle_order
                    vresult = (vcycle == img_nums_label)
                    if vresult:
                        vimg_level = vimg.split("_")[-1].split(".")[0]
                        self.vlabel_container.append(vimg_level)

                self.vlabel_container = [int(x) for x in self.vlabel_container]
                vdata1 = torch.from_numpy(self.vdata_container)

            vdatasum[...,ii] = vdata1
            ii += 1
        self.vdata = torch.from_numpy(vdatasum)
        self.vdata = self.vdata.permute(2, 1, 0)
        vlabelsum = np.array(self.vlabel_container)
        vlabelsum = torch.from_numpy(vlabelsum)
        self.vlabel = vlabelsum
        print("self.vlabel",self.vlabel.shape)
        self.vdataset_len = self.vdata.shape[0]
        print("self.vdataset_len", self.vdataset_len)



dataset = MyDadaSet()
# Data Loader for easy mini-batch return in training   # pipeline
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
vdataset = vMyDadaSet()
test_loader = torch.utils.data.DataLoader(dataset=vdataset, batch_size=BATCH_SIZE, shuffle=False)


class RNN(nn.Module):
    def __init__(self):
        # 找到Net的父类，将其转换为父类，再调用自己的 __init__()
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 5)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        #print("r_out",r_out.shape) #   ([22, 4, 64])
        # choose r_out at the last time step
        # Decode hidden state of last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()    # 交叉熵                   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    dataset.update_sampled_img()
    loss_all = []
    accuracy_all = []
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1,TIME_STEP, INPUT_SIZE)              # reshape x to (batch, time_step, input_size)
        output = rnn(b_x)  # output :float  shape: [22, 5] (batch_size,num_classes)
        # print("b_y",b_y) # integer
        b_y = b_y.to(torch.int64)
        loss = loss_func(output, b_y)             # cross entropy loss
        print("loss==",loss)
        loss_all.append(loss)
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        vdataset = vMyDadaSet()  # <__main__.MyDadaSet object at 0x0000022BEF5813C8>
        val_loader = DataLoader(vdataset, batch_size=9, shuffle=False, num_workers=0, )  # 1619968
        vdataset.vupdate_sampled_img()

        if step % 1 == 0:
            ii = 0
            total = 0
            correct = 0
            for vdt, vlb in val_loader:
                vdt = vdt.view(-1, TIME_STEP, INPUT_SIZE)
                test_output = rnn(vdt)                   # (samples, time_step, input_size)
                # print("test_output",test_output)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                print("pred_y",pred_y)
                total += vlb.size(0)
                print("vlb.data", vlb.data.numpy())
                correct += (pred_y == vlb.data.numpy()).sum()
            print("correct", correct)
            print("total", total)
            accuracy = 100*correct / total
            accuracy_all.append(accuracy)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# # print 10 predictions from test data
# test_output = rnn(test_x[:10].view(-1, 28, 28))
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')