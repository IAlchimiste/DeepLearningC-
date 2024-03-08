#include <iostream>
#include "NN.h"

int main() {
    srand(time(0));

    int numInputs = 2;
    int numOutputs = 2;
    int numHiddenLayers = 5;
    vector<float> hiddenLayer = {3, 2, 4, 7, 5};

    NeuralNetwork nn(numInputs, numHiddenLayers, hiddenLayer, numOutputs);
    
    vector<float> inputValues = {1, 0};
    vector<float> output = nn.feedForward(inputValues);

    cout << "Sortie non entrainée : " << endl;

    for (float &i : output) {
        cout << i << endl;
    }

    vector<float> expectedOutput = {1, 0};

    for (int i = 0; i< 500; i++) {
        nn.backPropagation(expectedOutput, output, 0.1);
        output = nn.feedForward(inputValues);
    }
    cout << "sortie souhaiter : " << endl;

    for (float &i : expectedOutput) {
        cout << i << endl;
    } 
    
    cout << "Sortie entrainée : " << endl;

    for (float &i : output) {
        cout << i << endl;
    }

    return 0;
}
