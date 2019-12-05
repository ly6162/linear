import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group = parser.add_argument_group('Data')
group.add_argument('-train', default="../data/learing_data/data_train.txt",help="train data")
group.add_argument('-teach',default="../data/learing_data/data_teacher.txt", help="teacher data")
group.add_argument('-save',default="../data", help="save a model to path")

group = parser.add_argument_group('train')
group.add_argument('-steps', default=1000,help="train all steps or num_epochs")
group.add_argument('-log_step', default=50,help="train all steps")
group.add_argument('-batch_size', default=100,help="train all steps")
group.add_argument('-learning_rate', type=float, default=0.001,help="Starting learning rate")

hparams = parser.parse_args()
