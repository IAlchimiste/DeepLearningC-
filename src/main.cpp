#include <iostream>
#include "NN.h"
#include <fstream>
#include <string>
#include <iomanip>

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

    int epoch = 20;
    
    int numInputs = 820;
    int numOutputs = 820;
    int numHiddenLayers = 2;
    vector<float> hiddenLayer = { 150, 75 };

    NeuralNetwork nn(numInputs, numHiddenLayers, hiddenLayer, numOutputs);

    // Show the neural network
    cout << "Neural Network : " << endl;
    for (int i = 0; i < nn.layers.size(); i++) {
        cout << "Layer " << i << " : " << endl;
        cout << "Neuron number : " << nn.layers[i].layer.size() << endl;
    }

    vector<float> output;

    vector<vector<float>> inputs;
    vector<vector<float>> desiredOutputs;

    cout << "Loading dataset from files." << endl;

    loadDataFromFile(inputs, desiredOutputs, numInputs, numOutputs);

    cout << "Start AI training." << endl;
    // afficher la date et l'heure actuelle
    time_t now = time(0);
    tm *ltm = localtime(&now);
    cout << "Start time : " << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << endl;

    int totalIterations = epoch * inputs.size();

    for (int j = 0; j < epoch; j++) {
        for (int i = 0; i < inputs.size(); i++) {
            output = nn.feedForward(inputs[i]);
            nn.backPropagation(desiredOutputs[i], nn.feedForward(inputs[i]), 0.1);

            // Calculate the current iteration
            int currentIteration = j * inputs.size() + i + 1;

            // Calculate the progress as a percentage
            float progress = (float)currentIteration / totalIterations;

            // Print the progress bar
            cout << "\r[";
            int pos = progress * 40;
            for (int p = 0; p < 40; ++p) {
                if (p < pos) cout << "=";
                else if (p == pos) cout << ">";
                else cout << " ";
            }
            cout << "] " << int(progress * 100.0) << " %\r" << flush;
        }
    }

    cout << endl << "AI trained." << endl;
    // afficher la date et l'heure actuelle
    now = time(0);
    ltm = localtime(&now);
    cout << "End time : " << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << endl;
    while (true) {
        string input;
        cout << "Enter a string : ";
        getline(cin, input);

        if (input == "exit") {
            break;
        }

        vector<float> inputVector = convertInput(input);
        output = nn.feedForward(inputVector);

        cout << "Output : " ;

        for (float &value : output) {
            char c = static_cast<char>(value * 255.0f);
            cout << c;
        }
        cout << endl;
    }
    return 0;
}
