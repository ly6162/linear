import numpy as np
import matplotlib.pyplot as plt
import json

def nn(x, w, b):
    return x * w + b

def eval_graph(framework,input,weight,bias):

    print("intpu: %s"%input)
    x = np.loadtxt('../data/data_train.txt')
    t = np.loadtxt('../data/data_teacher.txt')
    def fun(x):
        y=x*weight+bias
        return y

    out=fun(input)

    print("output:%s "%out)
    xx = np.linspace(0,2)

    #plt.plot(xx,xx,"b-",label="reference")

    y = nn(x, weight, bias)
    plt.plot(x, y, label="model")

    def f(x):
        return x * 2
    plt.plot(x, t, 'bo', label='teacher data')
    # Plot the initial line
    plt.plot([0, 1], [f(0), f(1)], 'r-', label='reference')
    plt.xlabel('$x$', fontsize=15)
    plt.ylabel('$t$', fontsize=15)
    plt.ylim([0, 2])
    plt.title('%s :line (x) vs eval (t)'%framework)
    plt.grid()
    plt.legend(loc=2)


    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([-0, 2, -0, 2]);
    plt.plot(input, out, ':gs',label="output")
    plt.legend()    # 各グラフの説明
    plt.show()

if __name__ == "__main__":
    model_path="../model_ML/ML_model.txt"
    with open(model_path, "r") as f:
        model = json.load(f)

    weight = model["weight"]
    bias = model["bias"]
    eval_graph("machine learning",0.1,weight,bias)