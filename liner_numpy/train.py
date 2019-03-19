import numpy as np
import matplotlib.pyplot as plt
import json
import time

"""
目的：機械学習の考え方を理解するため、線形モデルを用いてpythonで実現したプログラム
特徴：学習プロセスをグラフィックで出力しながら、データの収束を可視化していること

"""
model_path="../data/model_numpy/numpy_model.txt"

def _line(x):
    return  x * 2 + 0.5

def init():
    # Define the vector of input samples as x, with 20 values sampled from a uniform distribution
    # between 0 and 1
    x = np.loadtxt('../data/learing_data/data_train.txt')
    #x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    t = np.loadtxt('../data/learing_data/data_teacher.txt')

    # Generate the target values t from x with small gaussian noise so the estimation won't
    # be perfect.
    # Define a function f that represents the line that generates t without noise
    def f(x): return  _line(x)

    for i in range(0, len(x)):
       print("x[%d], t[%d], y[%d] = %f, %f, %f" % (i, i, i, x[i], t[i], f(x[i])))
    print("----------------------------------------")
    return x, t

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

# define the update function delta w
def delta_wb(w_k, x, b, t, learning_rate):
    delta_w, delta_b = gradient(w_k, x, b, t)
    return learning_rate * delta_w.sum(), learning_rate * delta_b.sum()

def draw_graph(x, w, b, t,count):
    y = nn(x, w, b)
    plt.plot(x, t, 'o', label="dots")
    plt.plot(x, y, label="line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([-0.5, 2, -0.5, 2]);
    plt.title('linear regression: train of count :%s'%count)
    plt.legend()    # 各グラフの説明
    plt.show()

def draw_grad(x, w, b, t, wb_cost):
    ws = np.linspace(-1, 4, num=100)  # weight values
    cost_ws = np.vectorize(lambda w: cost(nn(x, w, b), t))(ws)  # cost for each weight in ws

 # Plot the first 2 gradient descent updates
    plt.plot(ws, cost_ws, 'r-')  # Plot the error curve
 # Plot the updates
    for i in range(0, len(wb_cost) - 2):
        w1, b1, c1 = wb_cost[i]
        w2, b2, c2 = wb_cost[i + 1]
        plt.plot(w1, c1, 'bo')
        plt.plot([w1, w2], [c1, c2], 'b-')
        #plt.text(w1, c1 + 0.5, '$w({})$'.format(i))
 # Show figure
    plt.xlabel('$w$', fontsize=15)
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Gradient descent updates plotted on cost function')
    plt.axis([-1.1, 4,-1, 40]);
    plt.grid()
    plt.show()
weight=0
def train():
    x, t = init()

    # Set the initial weight parameter
    w = 0.1 #2
    b = -0.1 #-.5
    # Set the learning rate
    learning_rate = 0.001
    # Start performing the gradient descent updates, and print the weights and cost:
    steps = 10000 # number of gradient descent updates

    wb_cost = [(w, b, cost(nn(x, w, b), t))] # List to store the weight,costs values
    #draw_graph(x, w, b, t,0)
    start = time.time()
    for i in range(steps):
        dw, db = delta_wb(w, x, b, t, learning_rate)  # Get the delta w update
        w = w - dw  # Update the current weight parameter
        b = b - db
        wb_cost.append((w, b, cost(nn(x, w, b), t)))  # Add weight,cost to list
        loss=str(wb_cost[i][2])
        if i%1000==0:
            print("step:%s"%i,'loss:%s'%loss)
            #draw_graph(x, w, b, t,i)

            weight=wb_cost[i][0]
            bias = wb_cost[i][1]
            dict={"weight":weight,"bias":bias}
            #print(dict)
            with open(model_path,"w") as f:
                json.dump(dict,f)
    print('train time: %.5f' % (time.time()-start))
    print('weight: %s bias: %s loss: %s' %(weight,bias,loss))
    #draw_grad(x, w, b, t, wb_cost)


if __name__ == "__main__":
    train()