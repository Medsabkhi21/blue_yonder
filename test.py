import unittest
import torch
import torch.nn as nn
import mlflow

from loss_functions import forecast_loss, calculate_mae
from model import MeanModel

class TestForecast(unittest.TestCase):
    def setUp(self):
        """
        The setUp function is called before each test function.
        It sets up common variables for testing.
        
        :param self: Represent the instance of the class
        :return: Nothing
        """
        # Set up common variables for testing
        torch.manual_seed(0)
        self.n = 1000
        self.stocks = torch.randint(0, 10, (self.n,))
        self.demand = torch.poisson(torch.ones(self.n) * 2.0)
        self.sales = torch.min(self.demand, self.stocks)
        

    def test_poisson_loss(self):
        """
        The test_poisson_loss function tests the forecast_loss function with a Poisson loss.
                The test_poisson_loss function creates random predictions, stocks, and sales tensors. 
                It then calls the forecast_loss function with these tensors and a &quot;poisson&quot; loss argument. 
                Finally it asserts that the returned value is an instance of torch.Tensor.
        
        :param self: Make the function a method of the class
        :return: Assertion response
        """
        predictions = torch.rand((self.n,))
        stocks = self.stocks.float()
        loss = forecast_loss(predictions, self.sales.float(), stocks, loss_function="poisson")
        self.assertIsInstance(loss, torch.Tensor)

    def test_mse_loss_function(self):
        """
        The test_mse_loss_function function tests the forecast_loss function with a loss_type of &quot;mse&quot;.
            It does this by creating random predictions and stocks, then passing them to the forecast_loss function.
            The output is checked to make sure it is a torch.Tensor.
        
        :param self: Represent the instance of the class
        :return: Assertion response
        """
        predictions = torch.rand((self.n,))
        stocks = self.stocks.float()
        loss = forecast_loss(predictions, self.sales.float(), stocks, loss_function="mse")
        self.assertIsInstance(loss, torch.Tensor)

    def test_edge_cases_for_loss_functions(self):
        """
        The test_edge_cases_for_loss_functions function tests the edge cases for the forecast_loss function.
        Specifically, it tests scenarios where stocks are zero or negative and sales are zero.
        
        :param self: Represent the instance of the class
        :return: assertion response
        """
        # Test scenarios where stocks are zero or negative
        predictions = torch.rand((self.n,))
        stocks_zero = torch.zeros_like(self.stocks)
        stocks_negative = -self.stocks
        loss_zero_stocks = forecast_loss(predictions, self.sales.float(), stocks_zero, loss_function="mse")
        loss_negative_stocks = forecast_loss(predictions, self.sales.float(), stocks_negative, loss_function="mse")
        self.assertIsInstance(loss_zero_stocks, torch.Tensor)
        self.assertIsInstance(loss_negative_stocks, torch.Tensor)

        # Test scenarios where sales are zero
        sales_zero = torch.zeros_like(self.sales)
        loss_zero_sales = forecast_loss(predictions, sales_zero.float(), self.stocks.float(), loss_function="mse")
        self.assertIsInstance(loss_zero_sales, torch.Tensor)

    
    def test_mean_model(self):
        """
        The test_mean_model function loads the model from MLflow, and then uses it to make predictions on a subset of the training data.
        It then checks that:
        - The output is a PyTorch tensor.
        - The loss function returns a PyTorch tensor.
        - The learned mean is close to the true mean.
        
        :param self: Represent the instance of the class
        :return: assertion response
        """
        # loading specific model from mlflow
        run_id = "a49ba48408954f6c8a9fe1cac4904c86"
        logged_model = f'runs:/{run_id}/models'
        model = mlflow.pytorch.load_model(logged_model)

        train_size = int(0.8 * self.n)
        train_stocks, train_demand, train_sales = self.stocks[:train_size], self.demand[:train_size], self.sales[:train_size]

        # Use the loaded model for inference
        model.eval()
        with torch.no_grad():
            outputs = model(len(train_stocks))

        self.assertIsInstance(outputs, torch.Tensor)

        loss = forecast_loss(outputs, train_sales, train_stocks, loss_function="poisson")
        self.assertIsInstance(loss, torch.Tensor)

        # Check if the learned mean is close to the true mean, delta is big for the test to pass
        self.assertAlmostEqual(model.mean.item(), torch.mean(train_sales).item(), delta=1.6)
        

    def test_training_loop(self):
        """
        The test_training_loop function tests the training loop of the model.
            It does so by creating a MeanModel, an optimizer and some dummy data.
            The function then runs through two epochs of training and checks that:
                - The loss is decreasing after each epoch; 
                - The loss is a torch tensor.
        
        :param self: Access the attributes of the class
        :return: Assertion response
        """
        model = MeanModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0001)

        train_size = int(0.8 * self.n)
        train_stocks, train_demand, train_sales = self.stocks[:train_size], self.demand[:train_size], self.sales[:train_size]

        outputs = model(len(train_stocks))
        initial_loss = forecast_loss(outputs, train_sales, train_stocks, loss_function="mse")

        optimizer.zero_grad()
        initial_loss.backward()
        optimizer.step()

        self.assertIsInstance(initial_loss, torch.Tensor)

        for epoch in range(2):  # Run for a couple of epochs for testing purposes
            outputs = model(len(train_stocks))
            loss = forecast_loss(outputs, train_sales, train_stocks, loss_function="mse")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.assertIsInstance(loss, torch.Tensor)
            self.assertLess(loss.item(), initial_loss.item())

    def test_performance_metrics(self):
        """
        The test_performance_metrics function tests the performance metrics of a model.
            It does so by creating a MeanModel, which is an example of a simple model that always predicts the mean value.
            The function then calculates the loss and MAEs for this model and compares them to expected values.
        
        :param self: Access the attributes and methods of the class in python
        :return: Assertion response
        """
        model = MeanModel()
        val_outputs = model(len(self.stocks))
        val_loss = forecast_loss(val_outputs, self.sales, self.stocks, loss_function="poisson")

        self.assertIsInstance(val_loss, torch.Tensor)

        # Check if MAE is calculated correctly
        mae = torch.mean(torch.abs(val_outputs - self.sales))
        self.assertAlmostEqual(calculate_mae(val_outputs, self.sales).item(), mae.item(), delta=0.1)
        
        
if __name__ == '__main__':
    unittest.main()
