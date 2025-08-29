

# NeuronsIRL – Python Codes for the JSON file

This is used to **export a JSON file containing the weights and biases of a trained neural network**, which will be used by the [NeuronsIRL webapp](https://github.com/marhosa/NeuronsIRL).  
The JSON allows the webapp’s engine (`nnEngine`) to convert a **28×28 pixel grid drawing** into a recognized digit.  

👉 The exported JSON file should be placed inside:  
[NeuronsIRL/src/nnEngine](https://github.com/marhosa/NeuronsIRL/tree/main/src/nnEngine)

---

## Files in this Repository  

### 1. `MNIST Number Data.csv`  
- Source: [MNIST Digit Recognizer (Kaggle)](https://www.kaggle.com/competitions/mnist-digit-recognizer/data?select=train.csv)  
- Contains **785 columns**:  
  - Column 0 → digit labels (0–9)  
  - Columns 1–784 → pixel intensities of 28×28 grayscale images  

---

### 2. `ModelBuilder.py`  
- [Source Code](https://github.com/marhosa/Pytorch-NeuronsIRL/blob/main/ModelBuilder.py)  
- Responsible for:  
  - Loading the **CSV dataset**  
  - Splitting into **training (80%)** and **testing (20%)** sets  
  - Defining a **feed-forward neural network** (`MyModelProMax`) with:  
    - Input layer: 784 nodes (28×28 pixels)  
    - Hidden Layer 1: 128 nodes  
    - Hidden Layer 2: 64 nodes  
    - Output Layer: 10 nodes (digits 0–9)  
  - Training the model with **Adam optimizer** and **CrossEntropyLoss**  
  - Evaluating test accuracy  
  - Saving the trained weights as `numberModel.pth`  

---

### 3. `Jsonify.py`  
- [Source Code](https://github.com/marhosa/Pytorch-NeuronsIRL/blob/main/Jsonify.py)  
- Responsible for:  
  - Loading the trained model (`numberModel.pth`)  
  - Extracting **weights and biases** from:  
    - Layer 1 → `fc1`  
    - Layer 2 → `fc2`  
    - Layer 3 → `fc3`  
  - Exporting everything into a **JSON file**: `NnWeightsAndBiases.json`  

The JSON structure looks like:  
```json
{
    "l1w": [...],  // Layer 1 weights
    "l1b": [...],  // Layer 1 biases
    "l2w": [...],  // Layer 2 weights
    "l2b": [...],  // Layer 2 biases
    "l3w": [...],  // Layer 3 weights
    "l3b": [...]   // Layer 3 biases
}
