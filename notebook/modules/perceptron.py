from tqdm import tqdm
import torch
import numpy as np
from torch.nn import Linear
from torch.nn import Sigmoid, LogSoftmax
from torch.optim import SGD
from torch.nn import BCELoss
from sklearn.metrics import accuracy_score

class perceptron(torch.nn.Module):
    def __init__(self, input_shape, bias=True):
        super(perceptron, self).__init__()
        self.linear = Linear(input_shape, 1, bias=True)
        self.sigmoid = Sigmoid()
        
        
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class multi_class_perceptron(torch.nn.Module):
    def __init__(self, input_shape, output_shape, bias=True):
        super(multi_class_perceptron, self).__init__()
        self.linear = Linear(input_shape, output_shape, bias=True)
        self.softmax = LogSoftmax(dim=1)
        
        
    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x        
    
def train(model, train_data, optim, criterion, epochs=10, test_data=None):

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for it, example in enumerate(train_data):
            optim.zero_grad()
            f, t = example
            X = torch.FloatTensor(f)
            y = torch.FloatTensor(t)
            output = model.forward(X)
            loss = criterion(output.view(-1), y)
            total_loss += loss.data.numpy()
            loss.backward()

            optim.step()

        if test_data:
                model.eval()
                y_pred = []
                y_true = []
                threshold = 0.5
                for f, t in test_data:
                    X = torch.FloatTensor(f)
                    y = torch.FloatTensor(t)
                    output = model.forward(X)
                    y_true.append(y.data.numpy()[0])
                    y_pred.append(output.data.numpy()[0])
                    
                y_pred = [int(p >= threshold) for p in y_pred]
                a = accuracy_score(y_true, y_pred)



        total_loss /= (it + 1)
        print("Epoch Loss: {:.2f}, Validation Accuracy: {:.2f}".format(total_loss, a))
    
    return model