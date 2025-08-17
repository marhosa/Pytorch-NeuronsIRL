'''
marhosa
This extracts the hidden layers of the Model into a JSON file
The JSON file will be added to the NeuronsIRL "nnEngine" Folder
'''
print(f'\nRunning Jsonify')
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


#### define the same model
class MyModelProMax(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        #this part is just linear algebra 
        # they represent weight matrix and biases for each layer
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
    
    def forward(self, x):
        #just linear transformations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) #output
        return x
    

#### setting the sizes of the hidden layers
input_size = 784
hidden1_size = 128
hidden2_size = 64
output_size = 10


#### initialize the model
model = MyModelProMax(input_size, hidden1_size, hidden2_size, output_size)


#### Load the already trained model
model.load_state_dict(torch.load("numberModel.pth"))
model.eval() #set to evaluation mode


#### Extract the weights and biases of each layer 

l1_weights = model.fc1.weight.detach().numpy()
l1_biases = model.fc1.bias.detach().numpy()

l2_weights = model.fc2.weight.detach().numpy()
l2_biases = model.fc2.bias.detach().numpy()

l3_weights = model.fc3.weight.detach().numpy()
l3_biases = model.fc3.bias.detach().numpy()


#### Finding out the shape of the matrices
print(f'\nL1 Weight Shape: {l1_weights.shape}')
print(f"L1 Bias Shape: {l1_biases.shape}")

print(f'\nL2 Weight Shape: {l2_weights.shape}')
print(f"L2 Bias Shape: {l2_biases.shape}")

print(f'\nL3 Weight Shape: {l3_weights.shape}')
print(f"L3 Bias Shape: {l3_biases.shape}\n")


#### Store everything in a dict for json conversion

WeightsAndBiases = {
    "l1w": l1_weights.tolist(),
    "l1b": l1_biases.tolist(),

    "l2w": l2_weights.tolist(),
    "l2b": l2_biases.tolist(),

    "l3w": l3_weights.tolist(),
    "l3b": l3_biases.tolist()
}


#### converting dict to json and exporting the json file
jsonfile = json.dumps(WeightsAndBiases, indent=5)
with open("NnWeightsAndBiases.json", "w") as f:
    f.write(jsonfile)
print('Saved as "NnWeightsAndBiases.json"\n')
# marhosa