import json
import utils

if __name__ == "__main__":
    model_path="../data/model_numpy/numpy_model.txt"
    with open(model_path, "r") as f:
        model = json.load(f)

    weight = model["weight"]
    bias = model["bias"]
    input=0.6
    utils.eval_graph("machine learning",input,weight,bias)
