import os
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

from hparam import hparams
import linear_Pytoch.linear_model as line
import utils

def set_data(tt):
    #load data of txt to numpy
    x_input,y_input= utils.loadData()
    #
    x = np.reshape(x_input, (-1, 1))
    y=np.reshape(y_input, (-1, 1))
    #Defined a tensor data
    x_data = tt.FloatTensor(x)
    y_data = tt.FloatTensor(y)
    return x_input,y_input,x_data,y_data

def save(model):
    path=os.path.join(hparams.save,"model_Pytorch")
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
    print("step:%s loss:%s" % (step,loss))
    return w ,b

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
    x,t,x_data,y_data=set_data(tt)
    # Training: forward, loss, backward, step
    # Training loop
    start = time.time()

    plt.ion()
    w,b=0,0
    for step in range(hparams.train_steps+1):

        # Forward pass
        y_pred = model.linear_model(x_data)
        loss=model.loss(y_pred,y_data)

        if step % hparams.valid_steps==0 or step==hparams.valid_steps:
            w,b= out_log(step,loss,model)

            # Visualization of learning process
            y = utils.liner(x, w, b)
            plt.plot(x, t, 'o', label="dots")
            plt.plot(x, y, label="line")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis([-0.5, 2, -0.5, 2])
            plt.title('linear regression for PyToch: train of step :%s' % step)
            plt.pause(0.5)
            plt.cla()

        # Zero gradients
        optimizer.zero_grad()
        #勾配を計算、逆伝播
        loss.backward()
        # update weights
        optimizer.step()
    plt.ioff()
    save(model)
    print('train time: %.5f' % (time.time()-start))
    print('weight: %s bias: %s loss: %s' %(w,b,loss))
    utils.draw_graph("Pytoch",x, utils.liner(x, w, b),w, b, t,step)

if __name__ == "__main__":
    train()

