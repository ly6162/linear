import torch
import torch.nn.functional as F
from hparam import hparams

#Defined class of model and use Pytorch function
class Model_torch(torch.nn.Module):
    def __init__(self):
        super(Model_torch, self).__init__()
        # One in and one out
        self.linear_model = torch.nn.Linear(1, 1)
        # Defined loss function
        self.criterion = torch.nn.MSELoss(size_average=False)
        # Defined optimizer
        self.optimizer = torch.optim.SGD(self.linear_model.parameters(), lr=hparams.learning_rate)
        self.name="nn"

    def forward(self, x):
        y_pred = self.linear_model(x)
        return y_pred

    def loss(self,x,y):
        fn = self.criterion(x, y)
        return fn

#Defined class model  of user-defined function
class Model_my(object):
    def __init__(self,tt):
        # 自動微分tensorの定義
        # requires_grad = True 自動微分対象を指定
        # requires_grad = Falseの場合、更新されない
        self.W = tt.FloatTensor([.0])
        self.W.requires_grad=True
        self.b = tt.FloatTensor([.0])
        self.b.requires_grad=True
        #勾配法
        self.optimizer= torch.optim.SGD((self.W, self.b), lr=hparams.learning_rate)
        self.name = "my"

    def linear_model(self,x):
        # 線形モデル計算式(計算グラフを構築)
        linear_model = self.W * x + self.b
        return linear_model

    def loss(self,x,y):
        #2乗誤差
        fn = torch.sum(torch.pow(x - y, 2))
        return fn