import torch
import pickle
from ML_liner import inference as eval
import matplotlib.pyplot as plt

from pytoch_liner import liner_model
# our model
model_path="../model_Pytoch/pytorch_liner_model.pt"

model=(torch.load(model_path))
weight=model["weight"].numpy()
bias=model["bias"].numpy()
def out_text():
    print("weight: %s bias: %s"%(weight,bias))
if __name__ == "__main__":
    out_text()
    eval.eval_graph("Pytoch",.1,weight,bias)
    #W: [1.9103621] b: [-0.06653841]