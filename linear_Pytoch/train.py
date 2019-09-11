import os
import io
import numpy as np
import time
import torch
from hparam import hparam
import data
import linear_Pytoch.linear_model as line

def set_data(tt):
    #load data of txt to numpy
    x_input,y_input= data.load()
    #
    x = np.reshape(x_input, (-1, 1))
    y=np.reshape(y_input, (-1, 1))
    #Defined a tensor data
    x_data = tt.FloatTensor(x)
    y_data = tt.FloatTensor(y)
    return x_data,y_data

def save(model):
    path=os.path.join(hparam.save,"model_Pytorch")
    if not os.path.exists(path):
        os.mkdir(path)
    model_path = os.path.join(path, "ckpt_pytorch_linear.pt")
    torch.save(model, model_path)
    print("save: "+model_path)

def out_log(step,loss,model):
    if model.name == "nn":
        loss, w, b = loss.item(),model.linear_model.weight.item(), model.linear_model.bias.item()
    elif model.name == "my":
        loss, w, b = loss.item(), model.W.data.item(), model.b.data.item()

    print("step:%s" % step, )
    print("loss:%s weights:%s bias:%s" % (loss, w, b))

def train():

    #torchのnnを使ってグラフを構築
    model = line.Model_torch()

    if torch.cuda.is_available():
        tt = torch.cuda
        model.cuda()

    else:
        tt = torch
        #torchのnnを使ってグラフを構築
        model = line.Model_torch()

    #torchのnnを使わずに、グラフを自定義
    #model=line.Model_my(tt)

    #最適化関数
    optimizer=model.optimizer
    x_data,y_data=set_data(tt)
    # Training: forward, loss, backward, step
    # Training loop
    start = time.time()
    for step in range(hparam.steps):
        # Forward pass
        y_pred = model.linear_model(x_data)
        loss=model.loss(y_pred,y_data)

        if step%hparam.log_step==0:
                out_log(step,loss,model)

        # Zero gradients
        optimizer.zero_grad()
        #勾配を計算、逆伝播
        loss.backward()
        # update weights
        optimizer.step()

    print('train time(sec): %.5f' % (time.time()-start))
    save(model)

if __name__ == "__main__":
    train()

