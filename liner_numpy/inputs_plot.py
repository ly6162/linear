import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('data_train.txt')
t = np.loadtxt('data_teacher.txt')

def load_data():
    plt.plot(x, t, 'bo', label='t')
    # Plot the initial line
    # plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
    plt.xlabel('$x$', fontsize=15)
    plt.ylabel('$t$', fontsize=15)
    plt.ylim([0, 2])
    plt.title('teacher data (t)')
    plt.grid()
    plt.legend(loc=2)
    plt.show()
def show_line():
    def f(x):
        return x * 2
    x = np.loadtxt('data_train.txt')
    t = np.loadtxt('data_teacher.txt')

    plt.plot(x, t, 'bo', label='t')
    # Plot the initial line
    plt.plot([0, 1], [f(0), f(1)], 'r-', label='line')
    plt.xlabel('$x$', fontsize=15)
    plt.ylabel('$t$', fontsize=15)
    plt.ylim([0, 2])
    plt.title('line (x) vs teacher (t)')
    plt.grid()
    plt.legend(loc=2)
    plt.show()

load_data()
show_line()