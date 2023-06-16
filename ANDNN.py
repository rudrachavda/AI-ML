import torch
import torch.nn as nn
import torch.optim as optim

#--------------------------------------------------------------- Training Data --------------------------------------------------------------
train_x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float) # Problem
train_y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float) # Answers
#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------- Nueral Network --------------------------------------------------------------
class ANDNet(nn.Module):
    def __init__(self):
        super(ANDNet, self).__init__()
        
        # Input layer (2 neurons) -> Hidden layer (4 neurons) -> Output layer (1 neuron)
        self.fc1 = nn.Linear(2, 4)  # (2) input --> (4) hidden
        self.fc2 = nn.Linear(4, 1)  # (4) hidden --> (1) output

        # Activation is a function that decides whether a neuron should be activated or not. 
        # It takes in the weighted sum of the inputs and produces an output that is then passed on to the next layer.
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Apply  activation to the output of the first layer
        x = torch.sigmoid(self.fc2(x))  # Apply  activation to the output of the second layer
        return x

#--------------------------------------------------------------------------------------------------------------------------------------------
#initialize 
net = ANDNet()
criterion = nn.MSELoss() # Loss: the penalty for a bad prediction (helps the NN know what is right and what is wrong)
optimizer = optim.SGD(net.parameters(), lr=0.1) # Optimizer: Algorithm that adjusts the weights and learning rates of the neural network

#--------------------------------------------------------------- Training Loop --------------------------------------------------------------
for epoch in range(11111):
    
    output = net(train_x)
    loss = criterion(output, train_y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0: #run one epoch at a time
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------- Testing  -------------------------------------------------------------------
while True:
    test_input = input("Enter first input for testing (0 or 1), or 'exit' to quit: ")
    if test_input.lower() == "exit":
        break
    test_input2 = input("Enter second input for testing (0 or 1): ")

    # Convert test input to a tensor
    test_input_tensor = torch.tensor([[float(test_input), float(test_input2)]], dtype=torch.float)

    # Test the network
    prediction = net(test_input_tensor)
    print(f"Input: [{test_input}, {test_input2}], Prediction: {prediction.item()}")
