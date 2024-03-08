#  DeepLearningCpp 

```
Application : Deeplearning C++ Neural Network (NN)
Description : Full editable NN.
Made by     : The IAlchimiste 
```

#  IA compilation 

```
gpp main.cpp -o ia
```

#  Execution 

In a Terminal :
```
./ia
```
#  Config file 

```
No config file (coming soon).
```

#  Usage 

# Needed Variable :

```
int numInputs = 2; // Number of inputs
int numOutputs = 2; // Number of outputs
int numHiddenLayers = 5; // Number of hidden layers
vector<float> hiddenLayer = {3, 2, 4, 7, 5}; // Neuron per hidden layer
```

# Create the NN :

```
NeuralNetwork nn(numInputs, numHiddenLayers, hiddenLayer, numOutputs);
```

# Set and send inputs value :

```
vector<float> inputValues = {1, 0}; // Set inputs values
vector<float> outputs = nn.feedForward(inputValues); // Execute NN and get the outputs
```

# Show NN outputs :

```
for (float &i : outputs) { // for outputs
    cout << i << endl; // show value
}
```

# Set and start learning :

```
vector<float> inputValues = {1, 0}; // Set inputs values
vector<float> expectedOutputs = {1, 0}; // Set expected outputs
vector<float> outputs; // Create outputs lists

for (int i = 0; i< 500; i++) { // make 500 iteration
    output = nn.feedForward(inputValues); // Load outputs
    nn.backPropagation(expectedOutputs, outputs, 0.1); // Train network with the values
}
```


