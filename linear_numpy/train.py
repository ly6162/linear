import json
import time

import matplotlib.pyplot as plt
import utils
from hparam import hparams

"""
目的：機械学習の考え方を理解するため、線形モデルを用いてpythonで実現したプログラム
特徴：学習プロセスをグラフィックで出力しながら、データの収束を可視化していること

"""
model_path="../data/model_numpy/numpy_model.txt"

def _line(x):
    return  x * 2 + 0.5

# Define the neural network function y = x * w + b
def nn(x, w, b):
    return x * w + b

# Define the cost function
def cost(y, t):
    return ((t - y)**2).sum()

# define the gradient function. Remember that y = nn(x, w) = x * w + b
def gradient(w, x, b, t):
    #y=(nn(x, w, b)
    return  x * (nn(x, w, b) - t), (nn(x, w, b) - t)

# define the update function delta w&b
def delta_wb(w_k, x, b, t, learning_rate):
    delta_w, delta_b = gradient(w_k, x, b, t)
    return learning_rate * delta_w.sum(), learning_rate * delta_b.sum()

def train():
    x, t = utils.loadData()
    # Set the initial weight parameter
    w = 0.1 #2
    b = -0.1 #-.5

    wb_cost = [(w, b, cost(nn(x, w, b), t))] # List to store the weight,costs values

    start = time.time()

    plt.axis([0, 100, 0, 1])
    plt.ion()

    for step in range(hparams.steps+1):
        time.sleep(0.001)
        dw, db = delta_wb(w, x, b, t, hparams.learning_rate)  # Get the delta w update
        w = w - dw  # Update the current weight parameter
        b = b - db
        wb_cost.append((w, b, cost(nn(x, w, b), t)))  # Add weight,cost to list
        loss=str(wb_cost[i][2])

        if step % hparams.log_step==0 or step == hparams.steps:
            print("step:%s"%i,'loss:%s'%loss)

            weight=wb_cost[i][0]
            bias = wb_cost[i][1]
            dict={"weight":weight,"bias":bias}

            #save model
            with open(model_path,"w") as f:
                json.dump(dict,f)

            #Visualization of learning process
            y = nn(x, w, b)
            plt.plot(x, t, 'o', label="dots")
            plt.plot(x, y, label="line")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis([-0.5, 2, -0.5, 2])
            plt.title('linear regression: train of step :%s' % step)
            plt.pause(0.1)
            plt.cla()

    plt.ioff()
    print('train time: %.5f' % (time.time()-start))
    print('weight: %s bias: %s loss: %s' %(weight,bias,loss))
    utils.draw_graph(x, nn(x, w, b),w, b, t,step)

if __name__ == "__main__":
    train()