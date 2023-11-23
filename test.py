import unittest
import torch
from file import forecast_loss, calculate_mae
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mlflow
class MeanModel(nn.Module):
   def __init__(self):
       super(MeanModel, self).__init__()
       self.mean = nn.Parameter(torch.randn(1))
   def forward(self, n):
       return self.mean * torch.ones(n)
   
class TestForecastLossFunctions(unittest.TestCase):
    def setUp(self):
        # Set up common variables for testing
        torch.manual_seed(0)
        self.n = 1000
        self.stocks = torch.randint(0, 10, (self.n,))
        self.demand = torch.poisson(torch.ones(self.n) * 2.0)
        self.sales = torch.min(self.demand, self.stocks)
        

    def test_poisson_loss(self):
        predictions = torch.rand((self.n,))
        stocks = self.stocks.float()
        loss = forecast_loss(predictions, self.sales.float(), stocks, loss_function="poisson")
        self.assertIsInstance(loss, torch.Tensor)

    def test_mse_loss_function(self):
        predictions = torch.rand((self.n,))
        stocks = self.stocks.float()
        loss = forecast_loss(predictions, self.sales.float(), stocks, loss_function="mse")
        self.assertIsInstance(loss, torch.Tensor)

    def test_edge_cases_for_loss_functions(self):
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
        # Assuming you have started an MLflow run and have the run_id
        run_id = "5d510c8fed1044c998e469bfae85d11d"

        # Load the model from MLflow
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

        # Check if the learned mean is close to the true mean
        self.assertAlmostEqual(model.mean.item(), torch.mean(train_sales).item(), delta=0.1)
        

    def test_training_loop(self):
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
        model = MeanModel()
        val_outputs = model(len(self.stocks))
        val_loss = forecast_loss(val_outputs, self.sales, self.stocks, loss_function="poisson")

        self.assertIsInstance(val_loss, torch.Tensor)

        # Check if MAE is calculated correctly
        mae = torch.mean(torch.abs(val_outputs - self.sales))
        self.assertAlmostEqual(calculate_mae(val_outputs, self.sales).item(), mae.item(), delta=0.1)
        
        
if __name__ == '__main__':
    unittest.main()
