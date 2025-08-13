import os
import pandas as pd 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


#use cpu if gpu not available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   .to(device)
##########################################



#### this loads the CSV data.
df = pd.read_csv("MNIST Number Data.csv")


#### split the features and labels
X = df.iloc[:,0].values # first column of the csv (labels)
y = df.iloc[:,1:].values # the rest of the cols of the csv (features)

labels_tensor = torch.tensor(X, dtype=torch.long) # labels will be long(ints)
features_tensor = torch.tensor(y / 255.0, dtype=torch.float32) # features will be long


#### Change the train-test data to an 80/20 split
sample_size = len(labels_tensor)
train_size = int(sample_size * 0.8)
test_size = sample_size - train_size

indices = torch.randperm(sample_size) #gives random permuations of ints

train_indices = indices[:train_size]
test_indices = indices[-test_size:]


#### get the train and test labels-features

labels_train = labels_tensor[train_indices]
features_train = features_tensor[train_indices]

labels_test = labels_tensor[test_indices]
features_test = features_tensor[test_indices]


#### bundling the features and labels then making the data loader

train_dataset = TensorDataset(features_train, labels_train)
test_dataset = TensorDataset(features_test, labels_test)

#data loaders
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle = False)


#### model itself
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
    
input_size = features_train.shape[1]  # number of features, 784 for MNIST flattened images
hidden1_size = 128
hidden2_size = 64
output_size = 10  # number of classes

model = MyModelProMax(input_size, hidden1_size, hidden2_size, output_size).to(device) #the model

    

#### loss function and optimizer
#used CrossEntropyLoss for classification
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001)


#### training the model
epochCount = 10
for epoch in range(epochCount):
    print(f'\n\nEpoch: {epoch + 1} out of {epochCount}')

    for i in range(train_size):
        x = features_train[i].unsqueeze(0).to(device)
        y = labels_train[i].unsqueeze(0).to(device)

        optimizer.zero_grad() # clear prev gradients
        yPrediction = model(x)
        loss = lossFunction(yPrediction, y) #compare and calc loss
        loss.backward() #backpropagation
        optimizer.step() #update weights

        if i % 500 == 0: # made it every 500 to not crash the cli
            print(f'Sample {i}, Loss: {loss.item():.3f}')
print(f'\n\n\n')


#### Test the model 
model.eval() #set model to evaluation mode
correctPredictionCount = 0

with torch.no_grad():
    for i in range(test_size):
        x = features_test[i].unsqueeze(0).to(device)
        y = labels_test[i].to(device)
        pred = model(x) #prediction
        predictedLabel = torch.argmax(pred, dim = 1).item()

        if y.item() == predictedLabel:
            correctPredictionCount += 1

acc = (correctPredictionCount / test_size )*100
print(f'[Test Report] \nAccuracy: {acc:.04f}% \nCorrect: {correctPredictionCount} \nTotal:  {test_size}')



#### save the model 
currentDirectory = os.path.dirname(os.path.abspath(__file__))
torch.save(model.state_dict(), "numberModel.pth")
print(f'\n\nModel Saved to {currentDirectory}')

