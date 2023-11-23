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
   def forecast_loss(predictions, sales, stocks):
    epsilon = 1e-10  # Small constant to avoid log(0)

    # Calculate the mean of the Poisson distribution based on predictions
    lambda_ = torch.exp(predictions)

    # Poisson loss formula
    poisson_loss = lambda_ - sales * torch.log(lambda_ + epsilon)

    # Mean of the Poisson loss across all examples
    poisson_loss = torch.mean(poisson_loss)

    # Exceeded stock penalty
    exceeded_stock_penalty = torch.mean(torch.max(predictions - stocks, torch.zeros_like(predictions)))

    # Combine Poisson loss and exceeded stock penalty
    loss = poisson_loss + exceeded_stock_penalty

    return loss

def mse_loss_function(predictions,sales,stocks):
    mse_loss = nn.functional.mse_loss(predictions, sales)
    exceeded_stock_penalty = torch.mean(torch.max(predictions - stocks, torch.zeros_like(predictions)))
    loss = mse_loss + exceeded_stock_penalty
    return loss


def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def forecast_loss(predictions, sales, stocks, loss_function="poisson"):
    if loss_function =="poisson":
        return poisson_loss(predictions,sales,stocks)
    else:
        #mse loss by default
        return mse_loss_function(predictions,sales,stocks)
    


class MeanModel(nn.Module):
   def __init__(self):
       super(MeanModel, self).__init__()
       self.mean = nn.Parameter(torch.randn(1))
   def forward(self, n):
       return self.mean * torch.ones(n)