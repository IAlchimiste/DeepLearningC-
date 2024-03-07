#include <iostream>
#include <vector>
#include <random>

using namespace std;

class Neuron {
public:
    vector<float> weights;
    int numInputs;
    float bias;

    Neuron(int numInputs) {
        weights.resize(numInputs);

        for (int i = 0; i < numInputs; i++) {
            weights[i] = (float)rand() / RAND_MAX;
        }

        bias = (float)rand() / RAND_MAX;
    }

    vector<float> getWeights() {
        return weights;
    }

    float getBias() {
        return bias;
    }

    void setWeights(vector<float> newWeights) {
        weights = newWeights;
    }

    void setBias(float newBias) {
        bias = newBias;
    }
};

class Layer {
public:
    vector<Neuron> layer;

    Layer(int numNeurons, int numInputNeuron) {
        for (int i = 0; i < numNeurons; i++) {
            layer.push_back(Neuron(numInputNeuron));
        }
    }

    Layer() {}
};

class NeuralNetwork {
public:
    vector<Layer> layers;

    NeuralNetwork(int numInputs, int numHiddenLayers, vector<float> hiddenLayer, int numOutputs) {
        layers.push_back(Layer(numInputs, 1));

        for (int i = 0; i < numHiddenLayers; i++) {
            if (i == 0) {
                layers.push_back(Layer(hiddenLayer.at(i), numInputs));
            } else {
                layers.push_back(Layer(hiddenLayer.at(i), hiddenLayer.at(i - 1)));
            }
        }

        layers.push_back(Layer(numOutputs, hiddenLayer.at(numHiddenLayers - 1)));
    }

    float sigmoid(float x) {
        return 1 / (1 + exp(-x));
    }

    vector<float> feedForward(vector<float> inputValues) {
        vector<float> activations = inputValues;

        // Propagation avant à travers chaque couche
        for (Layer &layer : layers) {
            vector<float> newActivations;

            // Pour chaque neurone dans la couche
            for (Neuron &neuron : layer.layer) {
                float activation = 0.0;

                // Pour chaque poids dans le neurone
                for (int i = 0; i < neuron.weights.size(); i++) {
                    activation += neuron.weights[i] * activations[i];
                }

                activation += neuron.bias;
                newActivations.push_back(sigmoid(activation));
            }

            activations = newActivations;
        }

        return activations;
    }

    float sigmoidDerivative(float x) {
        float sigmoidValue = sigmoid(x);
        return sigmoidValue * (1 - sigmoidValue);
    }

    void backPropagation(vector<float> expectedOutput, vector<float> output, float learningRate) {
        vector<float> errors;

        // Calculer l'erreur de sortie
        for (int i = 0; i < output.size(); i++) {
            errors.push_back(expectedOutput[i] - output[i]);
        }

        // Rétropropagation de l'erreur et mise à jour des poids et des biais
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer &layer = layers[i];
            vector<float> previousErrors = errors;
            errors.clear();

            for (int j = 0; j < layer.layer.size(); j++) {
                Neuron &neuron = layer.layer[j];
                float error = previousErrors[j];
                float delta = error * sigmoidDerivative(neuron.bias);

                // Mise à jour des poids
                for (float &weight : neuron.weights) {
                    float previousActivation = (i > 0) ? layers[i - 1].layer[j].bias : 1.0;
                    weight += learningRate * delta * previousActivation;
                }

                // Mise à jour du biais
                neuron.bias += learningRate * delta;

                // Calculer l'erreur pour la couche précédente
                for (int k = 0; k < neuron.weights.size(); k++) {
                    errors.push_back(neuron.weights[k] * delta);
                }
            }
        }
    }
};

int main() {
    srand(time(0));

    int numInputs = 2;
    int numOutputs = 2;
    int numHiddenLayers = 5;
    vector<float> hiddenLayer = {3, 2, 4, 7, 5};

    NeuralNetwork nn(numInputs, numHiddenLayers, hiddenLayer, numOutputs);

    vector<float> inputValues = {1, 0};

    vector<float> output = nn.feedForward(inputValues);

    for (float &i : output) {
        cout << i << endl;
    }

    vector<float> expectedOutput = {1, 0};

    for (int i = 0; i< 500; i++) {
        nn.backPropagation(expectedOutput, output, 0.1);
        output = nn.feedForward(inputValues);
    }
    

    for (float &i : output) {
        cout << i << endl;
    }

    return 0;
}