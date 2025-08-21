# Ex-01-Developing-a-Neural-Network-Regression-Model
Developing a Neural Network Regression Model
### AIM
To develop a neural network regression model for the given dataset.

### THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

### Neural Network Model
Include the neural network model diagram.

### DESIGN STEPS
STEP 1: Generate Dataset
Create input values from 1 to 50 and add random noise to introduce variations in output values .

STEP 2: Initialize the Neural Network Model
Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

STEP 3: Define Loss Function and Optimizer
Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

STEP 4: Train the Model
Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

STEP 5: Plot the Loss Curve
Track the loss function values across epochs to visualize convergence.

STEP 6: Visualize the Best-Fit Line
Plot the original dataset along with the learned linear model.

STEP 7: Make Predictions
Use the trained model to predict for a new input value .

### PROGRAM
Name: Harini P

Register Number: 212224230082
```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#Generate input(x) and output(y)
torch.manual_seed(71)
X=torch.linspace(1, 50, 50).reshape(-1, 1)
e=torch.randint(-8, 9, (50, 1),dtype=torch.float)
y=2 * X + 1 + e

#plot the original data
plt.scatter(X,y,color='black')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generate Data for Linear Regression')
plt.show()
#Define the Linear Model Cass
class Model(nn.Module):
  def _init_(self, in_features, out_features):
    super()._init_()
    self.linear=nn.Linear(in_features, out_features)
    #self.linear=nn.linear(1,1)

  def forward(self,x):
    return self.linear(x)

    #initialise the model
torch.manual_seed(59)
model=Model(1,1)
#print inital weights and bias
initial_weight= model.linear.weight.item()
initial_bias= model.linear.bias.item()
print("\nName: ")
print("Register No: ")
print(f'Initial weight: {initial_weight:.2f}, Initial bias:{initial_bias:.8f}\n')
#Define Loss Function and optimizer
loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.001)
#Train the Model
epochs=100
losses=[]

for epoch in range(1,epochs+1):
  optimizer.zero_grad()
  y_pred=model(X)
  loss=loss_function(y_pred,y)
  losses.append(loss.item())

  loss.backward()
  optimizer.step()
#print loss,weight,and bias for EVERY epoch
print(f'epoch: {epoch:2} loss: {loss.item():10.8f} '
      f'weight: {model.linear.weight.item():10.8f}'
      f'bias: {model.linear.bias.item():10.8f}')
#plot loss curve
plt.plot(range(epochs),losses,color='black')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()
#Final Weights and bias
final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()
print("\nName: Harini P ")
print("Register No: 212224230082 ")
print(f'\nFinal weight: {final_weight:.8f}, Final bias: {final_bias:.8f}')
#Best fit line calculation
x1=torch.tensor([X.min().item(),X.max().item()])
y1=x1*final_weight+final_bias
#plot original data & best-fit line
plt.scatter(X,y,label='Original Data')
plt.plot(x1,y1,'r',label='Best-Fit Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.show()
#prediction for x=120
x_new=torch.tensor([[120.0]])
y_new_pred=model(x_new).item()
print("\nName: Harini P")
print("Register No: 212224230082")
print(f'\nPrediction for x = 120: {y_new_pred:.8f}')
```



# Initialize the Model, Loss Function, and Optimizer
### Dataset Information
<img width="812" height="604" alt="Screenshot 2025-08-21 204422" src="https://github.com/user-attachments/assets/bfbeaaf5-3195-4d8e-9ce4-91c6a75da4c1" />



# OUTPUT
### Training Loss Vs Iteration Plot 
<img width="798" height="583" alt="Screenshot 2025-08-21 204434" src="https://github.com/user-attachments/assets/79476045-5894-4b7d-b01a-7af73d404456" />


### Best Fit line plot Include your plot here              
<img width="839" height="601" alt="Screenshot 2025-08-21 210519" src="https://github.com/user-attachments/assets/64c1540c-98da-498d-844c-d7036d43e4f8" />



### New Sample Data Prediction
<img width="447" height="173" alt="Screenshot 2025-08-21 204533" src="https://github.com/user-attachments/assets/28114254-c486-491d-a621-f3b98a97df55" />



# RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
