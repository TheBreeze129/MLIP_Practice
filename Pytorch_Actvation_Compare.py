from numpy.lib import function_base
import torch.nn.functional as F

x_data = Variable(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.tensor([[.0], [.0], [1.], [1.]]))

class Model(torch.nn.Module):
    def __init__(self, active_func):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.activation = active_func

    def forward(self, x):
        y_pred = self.activation(self.linear(x))
        return y_pred
    

funcs = [
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.ELU,
    torch.nn.SELU,
    torch.nn.PReLU,
    torch.nn.LeakyReLU,
    torch.nn.Hardtanh,
    torch.nn.Sigmoid,
    torch.nn.Tanh
]

loss_list = []


def train_for_func(active_func):
    model = Model(active_func)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss_list.append([])

    for epoch in range(10):
        y_pred = model(x_data)

        loss = criterion(y_pred, y_data)
        loss_list[-1].append(loss.detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for active_func in funcs:
    train_for_func(active_func())

thres = torch.nn.Threshold(0.1, 0.5)
train_for_func(thres)