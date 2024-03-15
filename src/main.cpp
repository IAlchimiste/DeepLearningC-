#include <iostream>
#include "NN.h"
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>

// Fonction pour écrire un vecteur de chaînes dans un fichier
void writeStringsToFile(const std::vector<std::string>& strings, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (const auto& str : strings) {
        file << str << std::endl;
    }

    file.close();
}


float getMseFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1.0f;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "MSE : ") {
            return std::stof(line.substr(6));
        }
    }

    std::cerr << "MSE not found in file: " << filename << std::endl;
    return -1.0f;
}

// Fonction pour soustraire dans valeur avec la plus grande moins la plus petite
float sousVal(float val1, float val2) {
    if (val1 > val2) {
        return val1 - val2;
    } else {
        return val2 - val1;
    }
}

bool isMseIncreasingOrStagnant(float oldMse, float mse) {
    return oldMse == mse || mse > oldMse || sousVal(oldMse, mse) < 0.01;
}

bool shouldReset(float oldMse, float mse) {
    return std::isnan(mse) || (isMseIncreasingOrStagnant(oldMse, mse) && oldMse != 0.0f) || mse > 1.0f;
}

void printProgressBar(float progress, float mse) {
    std::cout << "\r[";
    int pos = progress * 40;
    for (int p = 0; p < 40; ++p) {
        if (p < pos) std::cout << "=";
        else if (p == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(2) << progress * 100.0 << " % - ";
    
    // Get the current time
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::cout << std::put_time(std::localtime(&now_c), "%T") << " - mse : " << std::setprecision(std::numeric_limits<float>::max_digits10) << mse  << std::flush;
}

using namespace std;

// Fonction pour ouvrir un fichier et lire sont contenu pour le mettre dans un vecteur
vector<float> readFile(string filename, int numValue) {
    vector<float> data;
    ifstream file(filename);

    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            for (char& c : line) {
                float normalized = static_cast<float>(c) / 255.0f;
                data.push_back(normalized);
                if (data.size() == numValue) {
                    break;
                }
            }
        }
    } else {
        cout << "Impossible d'ouvrir le fichier" << endl;
    }

    // Si moins de caractères ont été lus que nécessaire, remplissez le reste avec des zéros
    while (data.size() < numValue) {
        data.push_back(0.0f);
    }

    return data;
}

// Fonction pour convertir l'entré en valeur entre 0 et 1
vector<float> convertInput(string input) {
    vector<float> data;

    for (char& c : input) {
        float normalized = static_cast<float>(c) / 255.0f;
        data.push_back(normalized);
    }

    return data;
}

// Fonction pour convertir la sortie en texte
string convertOutput(vector<float> output) {
    string text;

    for (float& f : output) {
        char c = static_cast<char>(f * 255.0f);
        text += c;
    }

    return text;
}

// Fonction pour charger plusieur fichier dans un vecteur
vector<vector<float>> loadFiles(vector<string> filenames, int numValue) {
    vector<vector<float>> data;

    for (string &filename : filenames) {
        data.push_back(readFile(filename, numValue));
    }

    return data;
}

vector<string> listFiles(string folder) {
    vector<string> filenames;
    int i = 1;

    while (true) {
        string filename = folder + "/" + to_string(i) + ".txt";
        ifstream ifile(filename);

        if (!ifile) {
            break;
        }

        filenames.push_back(filename);
        i++;
    }

    return filenames;
}

void loadDataFromFile(vector<vector<float>> &inputs, vector<vector<float>> &desiredOutputs, int numInputs, int numOutputs) {
    vector<string> listInputFile = listFiles("data/inputs");
    vector<string> listDesiredOutputFile = listFiles("data/outputs");

    cout << "Inputs  file number : " << listInputFile.size() << endl;
    cout << "Outputs file number : " << listDesiredOutputFile.size() << endl;

    inputs = loadFiles(listInputFile, numInputs);
    desiredOutputs = loadFiles(listDesiredOutputFile, numOutputs);
}

int main() {
    srand(time(NULL));

    int epoch = 10;
    
    int numInputs = 200;
    int numOutputs = 200;
    int numHiddenLayers;
    float learningRate = 0.01;
    vector<float> hiddenLayer;
    vector<int> activationFunctionVector;
    vector<string> nnStringVector;

    // remplire de valeur aléatoire le nombre de neurone dans chaque couche
    for (int i = 0; i < numHiddenLayers; ++i) {
        int numNeurons = rand() % 250 + 51; // Générer un nombre aléatoire entre 1 et 100 pour chaque couche
        hiddenLayer.push_back(numNeurons);
        activationFunctionVector.push_back(rand() % 2); // Générer un nombre aléatoire entre 0 et 1 pour chaque couche
    }

    // Ajouter des fonctions d'activation pour la couche d'entrée et de sortie
    activationFunctionVector.push_back(0); // Fonction d'activation pour la couche d'entrée
    for (int i = 0; i < numHiddenLayers; i++) {
        activationFunctionVector.push_back(rand() % 5); // Fonction d'activation pour chaque couche cachée
    }
    activationFunctionVector.push_back(0); // Fonction d'activation pour la couche de sortie

    // Tableau de chaînes représentant les fonctions d'activation
    vector<string> activationFunctions = {"sigmoid", "relu", "tanh", "linear", "leakyRelu"};
    
    numHiddenLayers = rand() % 20 + 1;
    for (int i = 0; i < numHiddenLayers; ++i) {
        int numNeurons = rand() % 250 + 51;
        hiddenLayer.push_back(numNeurons);
        activationFunctionVector.push_back(rand() % 5);
    }

    NeuralNetwork nn = NeuralNetwork(numInputs, numHiddenLayers, hiddenLayer, activationFunctionVector, numOutputs);
    
    std::ofstream currentNetworkFile("data/config/current_network.txt");

    currentNetworkFile << "Neural Network : " << std::endl;
    for (int i = 0; i < nn.layers.size(); i++) {
        currentNetworkFile << "Layer " << i << " : " << "Neuron number : " << nn.layers[i].layer.size() << " ";
        if (!nn.layers[i].layer.empty()) {
            int activationFunctionId = nn.layers[i].layer[0].ActivationFunction;
            currentNetworkFile << " " << activationFunctions[activationFunctionId] << std::endl;
        }
    }
    currentNetworkFile << "MSE : " << 0.0f << std::endl;

    currentNetworkFile.close();

    vector<float> output;

    vector<vector<float>> inputs;
    vector<vector<float>> desiredOutputs;

    cout << "Loading dataset from files." << endl;

    loadDataFromFile(inputs, desiredOutputs, numInputs, numOutputs);

    cout << "Start AI training." << endl;

    int totalIterations = epoch * inputs.size();

    float totalError = 0.0f; // Pour stocker l'erreur totale
    float mse = 0.2f; // Pour stocker l'erreur quadratique moyenne
    float oldMse = 0.0f;
    
    while (mse > 0.01f) {
        mse = 0.0f;
        for (int j = 0; j < epoch; j++) {
            for (int i = 0; i < inputs.size(); i++) {

                // Calculate the current iteration
                int currentIteration = j * inputs.size() + i + 1;

                // Calculate the progress as a percentage
                float progress = (float)currentIteration / (epoch * inputs.size());

                // Print the progress bar with time
                printProgressBar(progress, mse);

                // Feed forward
                vector<float> output = nn.feedForward(inputs[i]);

                // Backpropagation
                nn.backPropagation(desiredOutputs[i], output, learningRate);

                // Calculate the error
                for (int k = 0; k < output.size(); k++) {
                    float error = desiredOutputs[i][k] - output[k];
                    totalError += error * error;
                }
            }
            // Calculate the mean squared error for this epoch
            mse = totalError / (inputs.size() * desiredOutputs[0].size());
            totalError = 0.0f;
        }
        cout << endl;
        float Bestmse = getMseFromFile("data/config/best_network.txt");
        if (mse > Bestmse || Bestmse == -1.0f) {
            
            std::ofstream bestNetworkFile("data/config/best_network.txt");
                
            bestNetworkFile << "Neural Network : " << std::endl;
            for (int i = 0; i < nn.layers.size(); i++) {
                bestNetworkFile << "Layer " << i << " : " << "Neuron number : " << nn.layers[i].layer.size() << " ";
                if (!nn.layers[i].layer.empty()) {
                    int activationFunctionId = nn.layers[i].layer[0].ActivationFunction;
                    bestNetworkFile << " " << activationFunctions[activationFunctionId] << std::endl;
                }
            }
            bestNetworkFile << "MSE : " << mse << std::endl;

            bestNetworkFile.close();
        }
        
        if ((shouldReset(oldMse, mse))) {
            cout << "TRAINING LOOK BAD !." << endl;

            std::ofstream previousNetworkFile("data/config/previous_network.txt");

            previousNetworkFile << "Neural Network : " << std::endl;
            for (int i = 0; i < nn.layers.size(); i++) {
                previousNetworkFile << "Layer " << i << " : " << "Neuron number : " << nn.layers[i].layer.size() << " ";
                if (!nn.layers[i].layer.empty()) {
                    int activationFunctionId = nn.layers[i].layer[0].ActivationFunction;
                    previousNetworkFile << " " << activationFunctions[activationFunctionId] << std::endl;
                }
            }
            previousNetworkFile << "MSE : " << mse << std::endl;

            previousNetworkFile.close();

            cout << "Resetting the network." << endl;
            // Reset network
            numHiddenLayers = rand() % 20 + 1;
            for (int i = 0; i < numHiddenLayers; ++i) {
                int numNeurons = rand() % 250 + 51;
                hiddenLayer.push_back(numNeurons);
                activationFunctionVector.push_back(rand() % 5);
            }
            nn = NeuralNetwork(numInputs, numHiddenLayers, hiddenLayer, activationFunctionVector, numOutputs);

            std::ofstream currentNetworkFile("data/config/current_network.txt");

            currentNetworkFile << "Neural Network : " << std::endl;
            for (int i = 0; i < nn.layers.size(); i++) {
                currentNetworkFile << "Layer " << i << " : " << "Neuron number : " << nn.layers[i].layer.size() << " ";
                if (!nn.layers[i].layer.empty()) {
                    int activationFunctionId = nn.layers[i].layer[0].ActivationFunction;
                    currentNetworkFile << " " << activationFunctions[activationFunctionId] << std::endl;
                }
            }
            currentNetworkFile << "MSE : " << 0.0f << std::endl;

            currentNetworkFile.close();

        }
        oldMse = mse;
    }
    return 0;
}
