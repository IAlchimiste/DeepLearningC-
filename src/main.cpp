#include "neuralNetwork.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

struct Config {
    int NUM_LAYERS;
    int NUM_NEURONS_PER_LAYER;
    int NUM_OUTPUTS;
    int NUM_INPUTS;
    float LEARNING_RATE;
    int NUM_EPOCHS;
};

class NeNetUse {
public:
    static Config readConfig(const string& filename);
    static int NeNetUsing(int argc, char* argv[]);
};

Config readConfig(const string& filename) {
    Config config;
    ifstream configFile(filename);

    if (configFile.is_open()) {
        // Lire chaque valeur numérique et les stocker dans l'ordre attendu
        configFile >> config.NUM_LAYERS;
        configFile >> config.NUM_NEURONS_PER_LAYER;
        configFile >> config.NUM_OUTPUTS;
        configFile >> config.NUM_INPUTS;
        configFile >> config.LEARNING_RATE;
        configFile >> config.NUM_EPOCHS;

        configFile.close();
    } else {
        cerr << "Impossible d'ouvrir le fichier de configuration." << endl;
    }

    return config;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " -l [input args...]" << endl;
        return 1;
    }
    
    // Lire les paramètres à partir du fichier de configuration
    Config config = readConfig("config.txt");

    int NUM_LAYERS = config.NUM_LAYERS;
    int NUM_NEURONS_PER_LAYER = config.NUM_NEURONS_PER_LAYER;
    int NUM_OUTPUTS = config.NUM_OUTPUTS;
    int NUM_INPUTS = config.NUM_INPUTS;
    float LEARNING_RATE = config.LEARNING_RATE;
    int NUM_EPOCHS = config.NUM_EPOCHS;

    bool isLearningMode = false;
    
    if (string(argv[1]) == "-l") {
        isLearningMode = true;
    } else if (string(argv[1]) == "-r") {
        NeuralNetwork network(NUM_INPUTS, NUM_LAYERS, NUM_NEURONS_PER_LAYER, NUM_OUTPUTS, LEARNING_RATE);
        network.resetWeights(0.1f); // Réinitialiser les poids à 0.1
        network.saveToFile("weights.bin");
        return 0;
    }
    if (isLearningMode) {
        if (argc != NUM_INPUTS + NUM_OUTPUTS + 2) {
            cout << "Le nombre d'arguments est incorrect en mode apprentissage." << endl;
            return 1;
        }

        NeuralNetwork network(NUM_INPUTS, NUM_LAYERS, NUM_NEURONS_PER_LAYER, NUM_OUTPUTS, LEARNING_RATE);
        
        vector<float> inputs;
        vector<float> desiredOutput;
        
        network.loadFromFile("weights.bin");

        // Récupérer les arguments d'entrée et de sortie
        for (int i = 2; i <= NUM_INPUTS + 1; i++) {
            inputs.push_back(stof(argv[i]));
        }
        for (int i = NUM_INPUTS + 2; i <= NUM_INPUTS + NUM_OUTPUTS + 1; i++) {
            desiredOutput.push_back(stof(argv[i]));
        }
        
        vector<vector<float>> outputs(NUM_LAYERS, vector<float>(NUM_NEURONS_PER_LAYER + 1));

        for (int i = 0; i < NUM_EPOCHS; i++) {
            network.backpropagation(inputs, desiredOutput, outputs);
            
            // Mettre à jour la sortie finale à chaque époque
            network.feedforward(inputs, outputs);
        }

        network.printOutput(outputs);

        network.saveToFile("weights.bin");
    } else {
        if (argc != NUM_INPUTS + 1) {
            cout << "Le nombre d'arguments est incorrect en mode inference." << endl;
            return 1;
        }

        NeuralNetwork network(NUM_INPUTS, NUM_LAYERS, NUM_NEURONS_PER_LAYER, NUM_OUTPUTS, LEARNING_RATE);
        network.loadFromFile("weights.bin");

        vector<float> inputs;

        // Récupérer les arguments d'entrée
        for (int i = 1; i <= NUM_INPUTS; i++) {
            inputs.push_back(stof(argv[i]));
        }

        vector<vector<float>> outputs(NUM_LAYERS, vector<float>(NUM_NEURONS_PER_LAYER + 1));

        // Faire l'inférence
        network.feedforward(inputs, outputs);

        network.printOutput(outputs);
    }

    return 0;
}
