#include <vector>
#include <random>
#include <algorithm>

using namespace std;

class Neuron {
public:
    vector<float> weights;
    int ActivationFunction;
    int numInputs;
    float bias;

    Neuron(int numInputs, int ActivationFunction) : numInputs(numInputs), ActivationFunction(ActivationFunction) {
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

    Layer(int numNeurons, int numInputNeuron, int ActivationFunction) {
        for (int i = 0; i < numNeurons; i++) {
            layer.push_back(Neuron(numInputNeuron, ActivationFunction));
        }
    }

    Layer() {}
};

class NeuralNetwork {
public:
    vector<Layer> layers;
    
    NeuralNetwork(int numInputs, int numHiddenLayers, vector<float> hiddenLayer, vector<int> activationFunctionVector, int numOutputs) {
        layers.push_back(Layer(numInputs, 1, activationFunctionVector[0]));

        for (int i = 0; i < numHiddenLayers; i++) {
            if (i == 0) {
                layers.push_back(Layer(hiddenLayer.at(i), numInputs, activationFunctionVector.at(i)));
            } else {
                layers.push_back(Layer(hiddenLayer.at(i), hiddenLayer.at(i - 1), activationFunctionVector.at(i)));
            }
        }

        layers.push_back(Layer(numOutputs, hiddenLayer.at(numHiddenLayers - 1), activationFunctionVector.at(numHiddenLayers)));
    }

    float sigmoid(float x) {
        return 1 / (1 + exp(-x));
    }

    float sigmoidDerivative(float x) {
        float sigmoidValue = sigmoid(x);
        return sigmoidValue * (1 - sigmoidValue);
    }

    float relu(float x) {
        return max(0.0f, x);
    }

    float reluDerivative(float x) {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    // Tangente hyperbolique
    float tanhActivation(float x) {
        return tanh(x);
    }

    // Softmax
    // Note: Cette fonction suppose que 'x' est la sortie de la couche finale du réseau de neurones.
    vector<float> softmax(vector<float> x) {
        vector<float> output(x.size());
        float maxElement = *max_element(x.begin(), x.end());
        float sum = 0;

        for (int i = 0; i < x.size(); i++) {
            output[i] = exp(x[i] - maxElement);
            sum += output[i];
        }

        for (int i = 0; i < x.size(); i++) {
            output[i] /= sum;
        }

        return output;
    }

    // Fonction d'activation linéaire
    float linear(float x) {
        return x;
    }

    // Leaky ReLU
    float leakyRelu(float x) {
        return max(0.01f * x, x);
    }

    // Leaky ReLU Derivative
    float leakyReluDerivative(float x) {
        return x > 0.0f ? 1.0f : 0.01f;
    }

    // Modification de la fonction d'activation
    float activationFunction(float x, int functionType) {
        switch (functionType) {
            case 0: return sigmoid(x);
            case 1: return relu(x);
            case 2: return tanhActivation(x);
            case 3: return linear(x);
            case 4: return leakyRelu(x);
            // Add more cases as needed
            default: return sigmoid(x);
        }
    }

    // Modification de la fonction dérivée d'activation
    // Modification de la fonction dérivée d'activation
    float activationFunctionDerivative(float x, int functionType) {
        switch (functionType) {
            case 0: return sigmoidDerivative(x);
            case 1: return reluDerivative(x);
            case 2: return 1 - pow(tanhActivation(x), 2);
            case 3: return 1;
            case 4: return leakyReluDerivative(x);
            // Add more cases as needed
            default: return sigmoidDerivative(x);
        }
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
                newActivations.push_back(activationFunction(activation, neuron.ActivationFunction));
            }

            activations = newActivations;
        }

        return activations;
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
                float delta = error * activationFunctionDerivative(neuron.bias, neuron.ActivationFunction);

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