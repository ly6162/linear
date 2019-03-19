import numpy as np
import time
import torch

import torch.nn.functional as F

#環境：python=3.x ,pytorch=1.0

# 学習データと教師データを読み込み
model_path="../model_Pytoch/pytorch_liner_model.pt"

def load_data(tt):
    x_input = np.loadtxt('../data/data_train.txt',) #学習データ
    y_input = np.loadtxt('../data/data_teacher.txt')#教師データ
    x1 = np.reshape(x_input, (-1, 1))
    y1=np.reshape(y_input, (-1, 1))
    x_data = tt.FloatTensor(x1)
    y_data = tt.FloatTensor(y1)
    return x_data,y_data

class Model_torch(torch.nn.Module):
    def __init__(self):
        super(Model_torch, self).__init__()
        self.linear_model = torch.nn.Linear(1, 1) # One in and one out
        self.criterion = torch.nn.MSELoss(size_average=False)  # Defined loss function
        self.optimizer = torch.optim.SGD(self.linear_model.parameters(), lr=0.001)  # Defined optimizer
        self.name="nn"

    def forward(self, x):
        y_pred = self.linear_model(x)
        return y_pred

    def loss(self,x,y):
        fn = self.criterion(x, y)
        return fn

class Model_my(object):
    def __init__(self):
        # 自動微分tensorの定義
        # requires_grad = True 自動微分対象を指定      # requires_grad = Falseの場合、更新されない
        self.W = torch.tensor(0.5, requires_grad=True)
        self.b = torch.tensor(0.5, requires_grad=True)
        #勾配法
        self.optimizer= torch.optim.SGD((self.W, self.b), lr=0.001)
        self.name = "my"
    def linear_model(self,x):
        # 線形モデル計算式(計算グラフを構築)
        linear_model = self.W * x + self.b
        return linear_model

    def loss(self,x,y):
        #2乗誤差
        fn = torch.sum(torch.pow(x - y, 2))
        return fn

def save(model,loss):
    line={}
    if model.name == "nn":
        #GPUのtensorは直接にnumpyに変更できない,一応cpu変換必要
        model_cpu = model.cpu()
        line["weight"] = model_cpu.linear_model.weight.data.numpy()[0]
        line["bias"] = model_cpu.linear_model.bias.data.numpy()
        loss = loss.data.cpu().numpy()
    elif model.name=="my":

        line["weight"]=model.W.data.numpy()
        line["bias"]=model.b.data.numpy()

    print('line: %s loss: %s' %(line,loss))

    torch.save(line, model_path)
    print("save: "+model_path)

def train():

    if torch.cuda.is_available():
        model = Model_torch().cuda()
        tt = torch.cuda
    else:
        #torchのnnを使ってグラフを構築
        model = Model_torch()
        #torchのnnを使わずに、グラフを定義
        #model=Model_my()
        tt=torch
    #最適化関数
    optimizer=model.optimizer
    x_data,y_data=load_data(tt)
    # Training: forward, loss, backward, step
    # Training loop
    start = time.time()
    for epoch in range(7000):
        # Forward pass
        y_pred = model.linear_model(x_data)
        loss=model.loss(y_pred,y_data)

        if epoch%1000==0:
            if model.name=="nn":
                print(list(model.parameters()), "step:%s" % epoch, "loss:%s" % loss.data.cpu().numpy())
            elif model.name=="my":
                print(model.W,model.b,"step:%s"%epoch, "loss:%s"%loss)

        # Zero gradients
        optimizer.zero_grad()
        # # 勾配を計算、逆伝播
        loss.backward()
        # update weights
        optimizer.step()
    print('train time: %.5f' % (time.time()-start))
    save(model,loss)

if __name__ == "__main__":
    train()