import numpy as np
import os
from skimage import io
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("time is :",datetime.now())



class MyDadaSet(Dataset):
    def __init__(self, seq_len=4, out_len=1):
        self.dataset_len = 1
        self.data_root = '/home/maggie/下载/0405/0401/'
        self.seq_len = seq_len
        self.out_len = out_len
        # self.data_container = np.zeros((512, 512, 45956), dtype=np.float64)
        # self.label_container = np.zeros((512, 512, 11489), dtype=np.float64)
        self.data_container = np.zeros((32, 32, seq_len), dtype=np.float32)
        self.label_container = np.zeros((32, 32, out_len), dtype=np.float32)
        self.data = None
        self.label = None

    # class torch.utils.data.Dataset作用: (1) 创建数据集,有__getitem__(self, index)函数来根据索引序号获取图片和标签, 有__len__(self)函数来获取数据集的长度.
    #  batchsize 指转换后有几行，而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
    # 如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。
    # __getitem__ 可以让对象实现迭代功能，这样就可以使用for...in... 来迭代该对象了
    # __getitem__ 支持索引, 以便于使用 dataset[i] 可以 获取第i个样本(0索引)
    def __getitem__(self, index):
        dt = self.data[index, ...].view(-1, 1)   # 相当于resize(),用来转换size大小;多行拼接成一行
        lb = self.label[index, ...]
        return dt, lb

    #  __len__ 使用``len(dataset)`` 可以返回数据集的大小
    def __len__(self):
        return self.dataset_len

    def update_sampled_img(self):
        # total_datanums = 0
        # total_labelnums = 0
        # for i in range(765,1438):
        datasum = []
        labelsum = []
        for i in range(764,1563):
            filePath = self.data_root + str(i)
            totalLens = len([x for x in os.listdir(filePath)])
            # total_datanum = (totalLens-self.seq_len - self.out_len) * self.seq_len
            #total_labelnum = (totalLens - self.seq_len - self.out_len) * self.out_len
            # total_datanums += total_datanum
            # total_labelnums += total_labelnum
        # print("total_datanums",total_datanums)
        # print("total_labelnums", total_labelnums)
        # print("mmmm888")
            lens = totalLens - self.seq_len - self.out_len + 1
            for j in range(lens):
                for s in range(self.seq_len):
                    img = Image.open(filePath + f'/{i}_{j + s}.jpg')
                    # print("img00",img)
                    img = img.resize((32, 32))  # 双线性插值
                    # print("img",img)
                    img =np.array(img)
                    # img = io.imread(self.data_root + f'{fold_id}/{fold_id}_{img_id+s:03}.jpg')
                    # img = io.imread(filePath + f'/{i}_{j + s}.jpg')
                    self.data_container[..., s] = img / 255

                for s in range(self.out_len):
                    img = Image.open(filePath + f'/{i}_{j + self.seq_len + s}.jpg')
                    img = img.resize((32, 32))
                    img = np.array(img)
                    #img = io.imread(filePath + f'/{i}_{j + self.seq_len + s}.jpg')
                    self.label_container[..., s] = img / 255


                mask = np.sum(self.data_container > 0, axis=-1) > self.seq_len / 2  # mask.shape (1200,1200)
                # print("origina data_container", self.data_container)
                # print("origina data_container.shape", self.data_container.shape)  # shape(512,512,156)
                #mask1 = self.data_container[mask, :]
                #print("mask1", mask1)

                data1 = torch.from_numpy(self.data_container[mask, :])  # data.shape torch.Size([262144, 2])
                label1 = torch.from_numpy(self.label_container[mask, :])

                if datasum == []:
                    datatemp = []
                    datatemp.append(data1)
                    datasum = np.concatenate(datatemp)
                else:
                    data2 = []
                    data2.append(datasum)
                    data2.append(data1)
                    datasum = np.concatenate(data2)
                self.data = torch.from_numpy(datasum)

                if labelsum == []:
                    labeltemp = []
                    labeltemp.append(label1)
                    labelsum = np.concatenate(labeltemp)
                else:
                    label2 = []
                    label2.append(labelsum)
                    label2.append(label1)
                    labelsum = np.concatenate(label2)
                self.label = torch.from_numpy(labelsum)

        self.dataset_len = self.data.shape[0]
        # print("data", self.data)
        # print("data.shape", self.data.shape)
        # print("label", self.label)
        # print("label.shape", self.label.shape)
        print("self.dataset_len", self.dataset_len)
        print("mmmmmm")
        # data shape:  torch.Size([1437585, 12]) label shape:  torch.Size([1437585, 3]) 1437585是个不定值

class vMyDadaSet(Dataset):
     def __init__(self, seq_len=4, out_len=1):
        self.vdataset_len = 1
        self.data_root = '/home/maggie/下载/0405/0401/'
        self.seq_len = seq_len
        self.out_len = out_len
        self.vdata_container = np.zeros((32, 32, seq_len), dtype=np.float32)
        self.vlabel_container = np.zeros((32, 32, out_len), dtype=np.float32)
        self.vdata = None
        self.vlabel = None

        # class torch.utils.data.Dataset作用: (1) 创建数据集,有__getitem__(self, index)函数来根据索引序号获取图片和标签, 有__len__(self)函数来获取数据集的长度.
        #  batchsize 指转换后有几行，而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        # 如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。
        # __getitem__ 可以让对象实现迭代功能，这样就可以使用for...in... 来迭代该对象了
         # __getitem__ 支持索引, 以便于使用 dataset[i] 可以 获取第i个样本(0索引)
     def __getitem__(self, index):
        vdt = self.vdata[index, ...].view(-1, 1)  # 相当于resize(),用来转换size大小;多行拼接成一行
        vlb = self.vlabel[index, ...]
        return vdt, vlb

     def __len__(self):
         return self.vdataset_len

     def vupdate_sampled_img(self):
         vdatasum = []
         vlabelsum = []
         for i in range(1563, 1662):

             filePath = self.data_root + str(i)
             vtotalLens = len([x for x in os.listdir(filePath)])
            # total_datanum = (totalLens-self.seq_len - self.out_len) * self.seq_len
            #total_labelnum = (totalLens - self.seq_len - self.out_len) * self.out_len
            # total_datanums += total_datanum
            # total_labelnums += total_labelnum
        # print("total_datanums",total_datanums)
        # print("total_labelnums", total_labelnums)
        # print("mmmm888")
             vlens = vtotalLens - self.seq_len - self.out_len + 1
             for j in range(vlens):
                 for s in range(self.seq_len):
                     # img = io.imread(self.data_root + f'{fold_id}/{fold_id}_{img_id+s:03}.jpg')
                     img = Image.open(filePath + f'/{i}_{j + s}.jpg')
                     img = img.resize((32, 32))
                     img = np.array(img)
                     #img = io.imread(fi lePath + f'/{i}_{j + s}.jpg')
                     self.vdata_container[..., s] = img / 255

                 for s in range(self.out_len):
                     img = Image.open(filePath + f'/{i}_{j + self.seq_len + s}.jpg')
                     img = img.resize((32, 32))
                     img = np.array(img)
                     #img = io.imread(filePath + f'/{i}_{j + self.seq_len + s}.jpg')
                     self.vlabel_container[..., s] = img / 255


                 mask = np.sum(self.vdata_container > 0, axis=-1) > self.seq_len / 2  # mask.shape (1200,1200)
                 # print("origina data_container", self.vdata_container)
                 # print("origina data_container.shape", self.vdata_container.shape)  # shape(512,512,156)
                 # mask1 = self.vdata_container[mask, :]
                 # print("mask1", mask1)

                 vdata1 = torch.from_numpy(self.vdata_container[mask, :])  # data.shape torch.Size([262144, 2])
                 vlabel1 = torch.from_numpy(self.vlabel_container[mask, :])

                 if vdatasum == []:
                     vdatatemp = []
                     vdatatemp.append(vdata1)
                     vdatasum = np.concatenate(vdatatemp)

                 else:
                     vdata2 = []
                     vdata2.append(vdatasum)
                     vdata2.append(vdata1)
                     vdatasum = np.concatenate(vdata2)
                 self.vdata = torch.from_numpy(vdatasum)

                 if vlabelsum == []:
                     vlabeltemp = []
                     vlabeltemp.append(vlabel1)
                     vlabelsum = np.concatenate(vlabeltemp)
                 else:
                     vlabel2 = []
                     vlabel2.append(vlabelsum)
                     vlabel2.append(vlabel1)
                     vlabelsum = np.concatenate(vlabel2)
                 self.vlabel = torch.from_numpy(vlabelsum)
         self.vdataset_len = self.vdata.shape[0]
         print("maggie0909")
         print("self.vdataset_len", self.vdataset_len)
         print("mmmmmm2")

class Model(nn.Module):
    '''
在PyTorch中，LSTM期望其所有输入都是3D张量，其尺寸定义如下：

lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers)
input_dim =输入数量（20的维度可代表20个输入）
hidden_dim =隐藏状态的大小; 每个LSTM单元在每个时间步产生的输出数。
n_layers =隐藏LSTM图层的数量; 通常是1到3之间的值; 值为1表示每个LSTM单元具有一个隐藏状态。 其默认值为1。

    '''
    def __init__(self, out_len=1):
        super(Model, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=10, num_layers=1,)
        self.decoder = nn.LSTMCell(input_size=10, hidden_size=10)
        self.linear = nn.Linear(10, 1)
        self.out_len = out_len

    def forward(self, inputs):
        """
        :param inputs: (batch_size, seq_len, vec_len)
        :return:
        """
        batch_size = inputs.shape[0]     # 4096
        outputs = torch.zeros((batch_size, self.out_len), device=inputs.device)
        hide_out, (h, c) = self.encoder(inputs.permute(1, 0, 2))
        h = h[-1, ...]
        c = c[-1, ...]
        for i in range(self.out_len):
            cur_input = self.attention(hide_out, h)
            h, c = self.decoder(cur_input, hx=(h, c))
            outputs[:, i] = self.linear(h).view(-1)
        return outputs

    @staticmethod
    def attention(encoder_hide, cur_hide):
        dist = torch.sum(encoder_hide * cur_hide[None], dim=-1)
        wt = F.softmax(dist, dim=0)
        cur_input = torch.sum(wt[..., None] * encoder_hide, dim=0)
        return cur_input



def train(gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}')  # 得到GPU编号 0
    epochs = 100
    batch_size = 1682368    # 21 ?  10:103424

    # 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
    # # 创建一个 Dataset 对象
    '''
      DataLoader是一个比较重要的类，它为我们提供的常用操作有：
              batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)
      '''
    dataset = MyDadaSet() # <__main__.MyDadaSet object at 0x0000022BEF5813C8>
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,) # <torch.utils.data.dataloader.DataLoader object at 0x0000025521EA76D8>
    model_prefix = 'weights/maggie041403'
    os.makedirs(model_prefix, exist_ok=True)
    net = Model()  # net Model((encoder): LSTM(1, 50, num_layers=3),(decoder): LSTMCell(50, 50),(linear): Linear(in_features=50, out_features=1, bias=True))
    # 将网络和数据放到GPU上
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)


    '''
    若 reduce = False，那么 size_average 参数失效，直接返回向量形式的 loss，即batch中每个元素对应的loss.
    若 reduce = True，那么 loss 返回的是标量：
           如果 size_average = True，返回 loss.mean().
           如果 size_average = False，返回 loss.sum()
    '''
    # 计算loss  损失函数：描述我们模型的预测距离目标还有多远；
    criteria = nn.MSELoss()  # torch.nn.MSELoss(reduce =True,size_average = True)
    # loss_func = torch.nn.CrossEntropyLoss()
    # 反向传播方法  优化算法：用于更新权重
    opt = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=1e-5, )  # net.parameters() 构建好神经网络后，网络的参数都保存在parameters()函数当中
    #  利用余旋退火 来降低学习率评估
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, 20, eta_min=1e-5)

    epoch_totalloss = []
    vepoch_totalloss = []
    for epoch in range(epochs):
        net.train()
        lr_scheduler.step()   # change learning rate
        dataset.update_sampled_img()
        epoch_loss = 0.0
        print("now the epoch is ::",epoch)
        num1 = 0
        num2 = 0
        for dt, lb in train_loader:   # dt.shape :torch.Size([21,5,1]   lb.shape:  21*1
            num1 += 1
            dt, lb = dt.to(device), lb.to(device)
            # 前向传播求出预测的值
            out = net(dt)   # torch.Size([4096,1])
            print("the shape of out", out)
            loss = criteria(out, lb)  # tensor(2.9710e-06, device='cuda:0', grad_fn=<MseLossBackward>)  变的值
            epoch_loss += loss.item()  # dt.shape[0] :21
            # 回传并更新梯度
            opt.zero_grad()#  梯度初始化为零
            loss.backward()# 反向传播求梯度 computer gradients   收集一系列新的梯度
            # 更新所有参数
            opt.step()
        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), f'{model_prefix}/epoch_{epoch+1}.pth')  # 只保存网络中的参数 (速度快, 占内存少)

        vdataset = vMyDadaSet()  # <__main__.MyDadaSet object at 0x0000022BEF5813C8>
        val_loader = DataLoader(vdataset, batch_size=3378176, shuffle=False, num_workers=0, )
        net.eval()
        vdataset.vupdate_sampled_img()
        vepoch_loss = 0.0
        for vdt, vlb in val_loader:   # dt.shape :torch.Size([21,5,1]   lb.shape:  21*1
            vdt, vlb = vdt.to(device), vlb.to(device)
            # 前向传播求出预测的值
            #vout = net(vdt)   # torch.Size([4096,1])
            with torch.no_grad():
                vout = net(vdt)
            vloss = criteria(vout, vlb)  # tensor(2.9710e-06, device='cuda:0', grad_fn=<MseLossBackward>)  变的值
            vepoch_loss += vloss.item()   # dt.shape[0] :21
            num2 += 1

        print("num1",num1)
        print("num2", num2)
        print("maggie")
        epoch_losss = np.sqrt(epoch_loss /  num1 )
        vepoch_losss = np.sqrt(vepoch_loss / num2)  # 54272
        print(f'epoch: {epoch+1}, loss: {epoch_losss:.6f}')
        print(f'epoch: {epoch+1}, vloss: {vepoch_losss:.6f}')
        epoch_totalloss.append(epoch_losss)
        vepoch_totalloss.append(vepoch_losss)
        print("epoch_totalloss", epoch_totalloss)
        print("vepoch_totalloss", vepoch_totalloss)


    epoch_num = np.linspace(1, epochs, epochs)
    plt.plot(epoch_num, epoch_totalloss, color='red')
    plt.plot(epoch_num, vepoch_totalloss, linestyle="--", color="orange")
    plt.xlabel(u'epoch', fontproperties='SimHei', fontsize=14)
    plt.ylabel(u'epoch_loss', fontproperties='SimHei', fontsize=14)
    plt.legend(["train_loss", "val_loss"], loc="upper right")
    plt.show()
    print("maggie22")



def infer(gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}')
    data_root = 'G:/0401/'
    sv_path = 'weights/maggie0409Val02'
    os.makedirs(sv_path, exist_ok=True)
    net = Model()
    state_dict = torch.load('weights/maggie0410/epoch_100.pth')
    # print(state_dict.keys())
    net.load_state_dict(state_dict)
    net.eval()
    '''                                                                                                                                                                                                            
     for param in net.parameters():
        print(param)
        break
    '''
    net.to(device)
    seq_len = 4
    # fold_id = 1
    for fold_id in range(1500, 1540):
        img_nms = sorted(glob.glob(data_root + f'{fold_id}/*jpg'))[-seq_len:]
        data_container = torch.zeros((512, 512, seq_len), dtype=torch.float32)
        for s, img_nm in enumerate(img_nms):
            img = io.imread(img_nm)
            data_container[..., s] = torch.from_numpy(img / 255)
        data_container = data_container.reshape(-1, seq_len, 1)
        data_container = data_container.to(device)
        out_img = []
        with torch.set_grad_enabled(False):
            for i in range(512):
                out = net(data_container[i * 512:(i + 1) * 512, ...]).reshape(1, 512,1).cpu().detach().numpy() * 255
                out = np.ceil(out).astype(np.int16)
                out_img.append(out)
        out = np.concatenate(tuple(out_img), axis=0)

        print(out)
        for i in range(1):
            io.imsave(f'{sv_path}/{fold_id}_{43}.jpg', out[..., i])


if __name__ == '__main__':

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # '

    train()
    #infer()

#
# cuda_gpu = torch.cuda.is_available()
# if (cuda_gpu):
#     print("Great, you have a GPU!")
# else:
#     print("Life is short -- consider a GPU!")



