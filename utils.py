import numpy as np
import matplotlib.pyplot as plt
from hparam import hparams

# Define the neural network function y = x * w + b
def liner(x, w, b):
    return x * w + b

def eval_graph(framework,input,weight,bias):

    print("intpu: %s"%input)
    x, t = loadData()
    def fun(x):
        y=x*weight+bias
        return y

    out=fun(input)

    print("output:%s "%out)
    xx = np.linspace(0,2)

    #plt.plot(xx,xx,"b-",label="reference")

    y = liner(x, weight, bias)
    plt.plot(x, y, label="model")

    def f(x):
        return x * 2
    plt.plot(x, t, 'bo', label='teacher data')
    # Plot the initial line
    plt.plot([0, 1], [f(0), f(1)], 'r-', label='reference')
    plt.xlabel('$x$', fontsize=15)
    plt.ylabel('$t$', fontsize=15)
    plt.ylim([0, 2])
    plt.title('%s :liner (x) vs eval (t)'%framework)
    plt.grid()
    plt.legend(loc=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([-0, 2, -0, 2]);
    plt.plot(input, out, ':gs',label="output")
    plt.legend()    # 各グラフの説明
    plt.show()

def loadData():
    # train data
    x_input = np.loadtxt(hparams.train)
    # teacher data
    y_input = np.loadtxt(hparams.teach)
    return x_input,y_input

def draw_graph(framework,x,y, w, b, t,step):
    plt.plot(x, t, 'o', label="dots")
    plt.plot(x, y, label="line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([-0.5, 2, -0.5, 2]);
    plt.title('linear regression  for %s: train of step :%s'%(framework,step))
    plt.legend()    # 各グラフの説明
    plt.show()

def datagen():
    def f(x):
        return x * 2

    # Create the targets t with some gaussian noise
    noise_variance = 0.2  # Variance of the gaussian noise
    # Gaussian noise error for each sample in x
    #x = np.random.uniform(0, 1, 10)
    x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    noise = np.random.randn(x.shape[0]) * noise_variance

    t = f(x) + noise
    np.savetxt('data_train.txt', x)
    np.savetxt('data_teacher.txt', t)

    plt.plot(x, t, 'bo', label='t')
    # Plot the initial line
    #plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
    plt.xlabel('$x$', fontsize=15)
    plt.ylabel('$t$', fontsize=15)
    plt.ylim([0, 2])
    plt.title('inputs (x) vs targets (t)')
    plt.grid()
    plt.legend(loc=2)
    plt.show()
