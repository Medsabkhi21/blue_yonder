import torch
import torch.nn as nn


def poisson_loss(predictions:torch.Tensor, sales:torch.Tensor):
    """
    The poisson_loss function calculates the Poisson loss 
    
    :param predictions: Calculate the mean of the poisson distribution
    :param sales: Calculate the mean of the poisson distribution based on predictions
    :return: The calculated poisson loss
    
    """
    epsilon = 1e-10  # Small constant to avoid log(0)

    # Calculate the mean of the Poisson distribution based on predictions
    lambda_ = torch.exp(predictions)

    # Poisson loss formula
    poisson_loss = lambda_ - sales * torch.log(lambda_ + epsilon)

    # Mean of the Poisson loss across all examples
    poisson_loss = torch.mean(poisson_loss)
    
    return poisson_loss

def poisson_loss_with_penality(predictions:torch.Tensor, sales:torch.Tensor, stocks:torch.Tensor):
    """
    The poisson_loss_with_penality function calculates the Poisson loss and adds a penalty for exceeding stock.
    
    :param predictions: Calculate the mean of the poisson distribution
    :param sales: Calculate the mean of the poisson distribution based on predictions
    :param stocks: Calculate the exceeded stock penalty
    :return: The combined poisson loss and exceeded stock penalty
    """
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

def mse_loss_function(predictions:torch.Tensor,sales:torch.Tensor):
    """
    The mse_loss_function function takes in two arguments:
    predictions - a tensor of shape (batch_size, 1) containing the predicted sales for each store and date.
    sales - a tensor of shape (batch_size, 1) containing the actual sales for each store and date.
    The function returns mse_loss 
    
    :param predictions: Pass the predicted sales values from the model
    :param sales: Calculate the loss
    :return: The mean squared error loss between the predictions and sales
    """
    mse_loss = nn.functional.mse_loss(predictions, sales)
    return mse_loss


def calculate_mae(predictions:torch.Tensor, targets:torch.Tensor):
    """
    The calculate_mae function takes in two arguments:
        predictions - a tensor of predicted values
        targets - a tensor of target values
    
    :param predictions: Store the predictions made by the model
    :param targets: Pass the actual values of the target variable
    :return: The mean absolute error between the predictions and targets
    """
    return torch.mean(torch.abs(predictions - targets))

def forecast_loss(predictions:torch.Tensor, sales:torch.Tensor, stocks:torch.Tensor, loss_function:str="poisson"):
    """
    The forecast_loss function takes in a list of predictions, sales and stocks.
    It then calculates the loss using either the poisson or mse loss function.
    The default is to use the poisson_loss function.
    
    :param predictions: Calculate the loss function
    :param sales: Calculate the loss function
    :param stocks: Calculate the loss function
    :param loss_function: Select the loss function to be used
    :return: The loss value for a given set of predictions, sales and stocks
    """
    if loss_function =="poisson":
        return poisson_loss(predictions,sales)
    elif loss_function =="poisson_with_penality":
        return poisson_loss_with_penality(predictions,sales,stocks)
    elif loss_function=="mse":
        return mse_loss_function(predictions,sales)
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")


