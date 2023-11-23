import torch
import torch.nn as nn

   
class MeanModel(nn.Module):
    def __init__(self):
       """
       The __init__ function is called when the object is created.
       It initializes the parameters of the model, which are stored in a dictionary named self.named_parameters().
       The super() function allows us to call methods from parent classes (in this case, nn.Module). 
       
       
       :param self: Represent the instance of the class
       :return: Nothing
       """
       super(MeanModel, self).__init__()
       self.mean = nn.Parameter(torch.randn(1))
  
    def forward(self, n:int):
       """
       The forward function returns a tensor of size n with all elements equal to the mean.
       
       :param self: Access variables that belong to the class
       :param n: Specify the number of samples to draw
       :return: A tensor with the same size as n
       """
       return self.mean * torch.ones(n)
