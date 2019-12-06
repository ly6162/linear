import os
import torch
from hparam import hparams
import utils

def get_param():
    path = os.path.join(hparams.save, "model_Pytorch")
    model_path = os.path.join(path, "ckpt_pytorch_linear.pt")
    #load  model
    model = torch.load(model_path)

    if model.name == "nn":
        #GPUのtensorは直接にnumpyに変更できない,一応cpu変換必要
        # model_cpu = model.cpu()
        weight = model.linear_model.weight.data.numpy()[0]
        bias= model.linear_model.bias.data.numpy()
    elif model.name=="my":
        weight=model.W.data.numpy()
        bias=model.b.data.numpy()
    return weight,bias

if __name__ == "__main__":
    weight, bias=get_param()
    print("weight: %s bias: %s" % (weight, bias))
    utils.eval_graph("Pytoch", .1, weight, bias)