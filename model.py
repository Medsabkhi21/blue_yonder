import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,Dataset
import matplotlib.pyplot as plt
import mlflow

# Define the input tensors for stocks and sales
n = 1000
torch.manual_seed(0)
stocks = torch.randint(0, 10, (n,))
demand = torch.poisson(torch.ones(n) * 2.0)
sales = torch.min(demand, stocks)


def poisson_loss(predictions, sales, stocks):
   # Ensure that predictions do not exceed the stock
   predictions = torch.min(predictions, stocks)
   # Calculate the Poisson loss
   loss = torch.nn.functional.poisson_nll_loss(predictions, sales)
   return loss

def mse_loss_function(predictions,sales,stocks):
    mse_loss = nn.functional.mse_loss(predictions, sales)
    exceeded_stock_penalty = torch.mean(torch.max(predictions - stocks, torch.zeros_like(predictions)))
    loss = mse_loss + exceeded_stock_penalty
    return loss

class MeanModel(nn.Module):
   def __init__(self):
       super(MeanModel, self).__init__()
       self.mean = nn.Parameter(torch.randn(1))
   def forward(self, n):
       return self.mean * torch.ones(n)
   
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
