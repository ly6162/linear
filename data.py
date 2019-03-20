import hparam
import numpy as np
def load():
    # train data
    x_input = np.loadtxt(hparam.hparam.train)
    # teacher data
    y_input = np.loadtxt('../data/learing_data/data_teacher.txt')
    return x_input,y_input