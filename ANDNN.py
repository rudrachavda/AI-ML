import torch #import pytorch library
import torch.nn as nn
import torch.optim as optim

# Nueral Network learning is like learning a school course, You are trained on certain data, and then tested on it.

#--------------------------------------------------------------- Training Data --------------------------------------------------------------
train_x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float) # Problem
train_y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float) # Answers
#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------- Nueral Network --------------------------------------------------------------
class ANDNet(nn.Module): # defines a new class; ANDNet, inherits from the nn.Module class, base class provided by PyTorch 
    def __init__(self): #
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
#initialize NN module
net = ANDNet()
criterion = nn.MSELoss() # Loss: the penalty for a bad prediction (helps the NN know what is right and what is wrong)
optimizer = optim.SGD(net.parameters(), lr=0.1) # Optimizer: Algorithm that adjusts the weights and learning rates of the neural network

#--------------------------------------------------------------- Training Loop --------------------------------------------------------------
for epoch in range(20000): # runs 11111 times
    
    output = net(train_x) # Feeds training data from train_x into the NN by calling the network as a function "net"
    loss = criterion(output, train_y) #calculates the loss between the problem(train_x) and solutions(train_y)

    # This line clears (zeros out) the gradients of all the parameters in the neural network. Gradients are accumulated by default for each 
    # parameter during the backward pass, and calling zero_grad() ensures that the gradients are reset to zero before the next backward pass. 
    # This step is necessary because PyTorch accumulates gradients by default if you don't zero them out, the gradients from previous 
    # iterations will be accumulated with the current gradients, leading to incorrect parameter updates.
    
    optimizer.zero_grad() #prevents interference from pervious backpropogations
    
    loss.backward() #performs backpropogation; computes the gradient of the loss  with respect to each parameter, 
                    #indicating how much the loss will change when the parameter is adjusted
    
    optimizer.step() #This line updates the neural network parameters based on the gradients computed during the backward pass.

    if (epoch + 1) % 1 == 0: #run one epoch at a time
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------- Testing  -------------------------------------------------------------------
while True:
    print("")
    print("If both inputs are 1 --> 1, otherwise everything else should be 0")
    test_input = input("Enter first input for testing (0 or 1), or 'exit' to quit: ")
    if test_input.lower() == "exit":
        break
    test_input2 = input("Enter second input for testing (0 or 1): ")

    # Convert test input to a tensor
    test_input_tensor = torch.tensor([[float(test_input), float(test_input2)]], dtype=torch.float)

    # Test the network
    prediction = net(test_input_tensor)
    print(f"Input: [{test_input}, {test_input2}], Prediction: {prediction.item()}")
