#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

class NeuralNetwork {
private:
    int NUM_OUTPUTS;  // Nombre de sorties du réseau
    float LEARNING_RATE;  // Taux d'apprentissage du réseau
    int numInputs;  // Nombre d'entrées du réseau
    int numLayers;  // Nombre de couches du réseau
    int numNeuronsPerLayer;  // Nombre de neurones par couche
    vector<vector<vector<float>>> weights;  // Poids des connexions entre les neurones
    vector<vector<float>> delta;  // Erreurs de sortie des neurones


public:
    NeuralNetwork(int _numInputs, int _numLayers, int _numNeuronsPerLayer, int _NUM_OUTPUTS, float _LEARNING_RATE)
        : NUM_OUTPUTS(_NUM_OUTPUTS), LEARNING_RATE(_LEARNING_RATE),
          numInputs(_numInputs), numLayers(_numLayers), numNeuronsPerLayer(_numNeuronsPerLayer) {
        // Initialisation des vecteurs weights et delta avec la taille appropriée
        weights.resize(numLayers - 1);
        delta.resize(numLayers, vector<float>(numNeuronsPerLayer + 1));

        for (int l = 0; l < numLayers - 1; l++) {
            weights[l].resize(numNeuronsPerLayer);
            for (int j = 0; j < numNeuronsPerLayer; j++) {
                weights[l][j].resize(numNeuronsPerLayer + 1, 0.1f);
            }
        }
    }

    float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    void feedforward(const vector<float>& inputs, vector<vector<float>>& outputs) {
        int numInputs = this->numInputs;
        int numLayers = this->numLayers;
        int numNeuronsPerLayer = this->numNeuronsPerLayer;

        for (int i = 0; i < numInputs; i++) {
            outputs[0][i] = inputs[i];
        }
        outputs[0][numInputs] = 1;

        for (int l = 1; l < numLayers; l++) {
            for (int j = 0; j < numNeuronsPerLayer; j++) {
                float sum = 0.0f;
                for (int i = 0; i <= numNeuronsPerLayer; i++) {
                    sum += weights[l - 1][j][i] * outputs[l - 1][i];
                }
                outputs[l][j] = sigmoid(sum);
            }
            outputs[l][numNeuronsPerLayer] = 1;
        }
    }

    void backpropagation(const vector<float>& inputs, const vector<float>& desiredOutput, vector<vector<float>>& outputs) {
        int numLayers = this->numLayers;
        int numNeuronsPerLayer = this->numNeuronsPerLayer;
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            delta[numLayers - 1][j] = (desiredOutput[j] - outputs[numLayers - 1][j]) * outputs[numLayers - 1][j] * (1 - outputs[numLayers - 1][j]);
        }
        for (int l = numLayers - 2; l >= 0; l--) {
            for (int j = 0; j < numNeuronsPerLayer; j++) {
                float error = 0.0f;
                for (int k = 0; k < NUM_OUTPUTS; k++) {
                    error += delta[l + 1][k] * weights[l][k][j];
                }
                delta[l][j] = error * outputs[l][j] * (1 - outputs[l][j]);
            }
        }
        for (int l = numLayers - 2; l >= 0; l--) {
            for (int j = 0; j < numNeuronsPerLayer; j++) {
                for (int i = 0; i <= numNeuronsPerLayer; i++) {
                    weights[l][j][i] += LEARNING_RATE * delta[l + 1][j] * outputs[l][i];
                }
            }
        }
    }

    void saveToFile(const string& filename) const {
        ofstream file(filename, ios::binary);
        if (file) {
            int numInputs = this->numInputs;
            int numLayers = this->numLayers;
            int numNeuronsPerLayer = this->numNeuronsPerLayer;

            file.write(reinterpret_cast<const char*>(&numInputs), sizeof(numInputs));
            file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));
            file.write(reinterpret_cast<const char*>(&numNeuronsPerLayer), sizeof(numNeuronsPerLayer));

            for (int l = 0; l < numLayers - 1; l++) {
                for (int j = 0; j < numNeuronsPerLayer; j++) {
                    file.write(reinterpret_cast<const char*>(weights[l][j].data()), (numNeuronsPerLayer + 1) * sizeof(float));
                }
            }

            file.close();
        } else {
            cerr << "Failed to open file for writing: " << filename << endl;
        }
    }

    void loadFromFile(const string& filename) {
        ifstream file(filename, ios::binary);
        if (file) {
            int numInputs, numLayers, numNeuronsPerLayer;
            file.read(reinterpret_cast<char*>(&numInputs), sizeof(numInputs));
            file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
            file.read(reinterpret_cast<char*>(&numNeuronsPerLayer), sizeof(numNeuronsPerLayer));

            weights.resize(numLayers - 1);
            delta.resize(numLayers, vector<float>(numNeuronsPerLayer + 1));

            for (int l = 0; l < numLayers - 1; l++) {
                weights[l].resize(numNeuronsPerLayer);
                for (int j = 0; j < numNeuronsPerLayer; j++) {
                    weights[l][j].resize(numNeuronsPerLayer + 1);
                    file.read(reinterpret_cast<char*>(weights[l][j].data()), (numNeuronsPerLayer + 1) * sizeof(float));
                }
            }

            file.close();
        } else {
            cerr << "Failed to open file for reading: " << filename << endl;
        }
    }
    void printOutput(const vector<vector<float>>& outputs) {
        int numLayers = outputs.size();
        int lastLayerIndex = numLayers - 1;

        for (int i = 0; i < NUM_OUTPUTS; i++) {
            cout << outputs[lastLayerIndex][i] << " ";
        }
        cout << endl;
    }
    void resetWeights(float initialValue) {
        for (int l = 0; l < numLayers - 1; l++) {
            for (int j = 0; j < numNeuronsPerLayer; j++) {
                for (int i = 0; i <= numNeuronsPerLayer; i++) {
                    weights[l][j][i] = initialValue;
                }
            }
        }
    }
};