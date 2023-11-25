import torch
import torch.nn as nn


def poisson_loss(predictions:torch.Tensor, demand:torch.Tensor):
    """
    The poisson_loss function calculates the Poisson loss 
    
    :param predictions: pytorch tensor of the demand predictions.
    :param demand: pytorch tensor of the demand values
    :return: The calculated poisson loss
    
    """
    if torch.any(demand <= 0):
        raise ValueError("Demand values must be positive.")

    epsilon = 1e-10  # Small constant to avoid log(0)

    # Calculate the mean of the Poisson distribution based on predictions
    lambda_ = torch.exp(predictions)

    # Poisson loss formula
    poisson_loss = lambda_ - demand * torch.log(lambda_ + epsilon)

    # Mean of the Poisson loss across all examples
    poisson_loss = torch.mean(poisson_loss)
    
    return poisson_loss

def poisson_loss_with_penality(predictions:torch.Tensor, demand:torch.Tensor, stocks:torch.Tensor):
    """
    The poisson_loss_with_penality function calculates the Poisson loss and adds a penalty for exceeding stock.
    
    :param predictions: Calculate the mean of the poisson distribution
    :param demand: Calculate the mean of the poisson distribution based on predictions
    :param stocks: Calculate the exceeded stock penalty
    :return: The combined poisson loss and exceeded stock penalty
    """
    if torch.any(demand <= 0):
        raise ValueError("Demand values must be positive.")
    
    epsilon = 1e-10  # Small constant to avoid log(0)

    # Calculate the mean of the Poisson distribution based on predictions
    lambda_ = torch.exp(predictions)

    # Poisson loss formula
    poisson_loss = lambda_ - demand * torch.log(lambda_ + epsilon)

    # Mean of the Poisson loss across all examples
    poisson_loss = torch.mean(poisson_loss)

    # Exceeded stock penalty
    exceeded_stock_penalty = torch.mean(torch.max(predictions - stocks, torch.zeros_like(predictions)))

    # Combine Poisson loss and exceeded stock penalty
    loss = poisson_loss + exceeded_stock_penalty

    return loss

def mse_loss_function(predictions:torch.Tensor,demand:torch.Tensor):
    """
    The mse_loss_function function takes in two arguments:
    predictions - a tensor of shape (batch_size, 1) containing the predicted sales for each store and date.
    demand - a tensor of shape (batch_size, 1) containing the actual demand for each store and date.
    The function returns mse_loss 
    
    :param predictions: Pass the predicted sales  values from the model
    :param demand: pytorch Tensor of the values
    :return: The mean squared error loss between the predictions and demand
    """
    if torch.any(demand <= 0):
        raise ValueError("Demand values must be positive ")
    mse_loss = nn.functional.mse_loss(predictions, demand)
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

def forecast_loss(predictions:torch.Tensor, demand:torch.Tensor, stocks:torch.Tensor, loss_function:str="poisson"):
    """
    The forecast_loss function takes in a list of predictions, demand and stocks.
    It then calculates the loss using either the poisson or mse loss function.
    The default is to use the poisson_loss function.
    
    :param predictions: Calculate the loss function
    :param demand: pytorch tensor of demand values
    :param stocks: pytorch tensor of stocks values
    :param loss_function: Select the loss function to be used
    :return: The loss value for a given set of predictions, sales and stocks
    """
    if torch.any(demand < 0.):
        raise ValueError("Demand values must be positive")
    
    if loss_function =="poisson":
        return poisson_loss(predictions,demand)
    elif loss_function =="poisson_with_penality": #just for testing this is not the correct solution.
        return poisson_loss_with_penality(predictions,demand,stocks)
    elif loss_function=="mse":
        return mse_loss_function(predictions,demand)
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")


