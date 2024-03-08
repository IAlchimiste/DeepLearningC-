#include <iostream>
#include "NN.h"
#include <fstream>
#include <string>

using namespace std;

// Fonction pour ouvrir un fichier et lire sont contenu pour le mettre dans un vecteur
vector<float> readFile(string filename) {
    vector<float> data;
    ifstream file(filename);

    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            data.push_back(stof(line));
        }
    } else {
        cout << "Impossible d'ouvrir le fichier" << endl;
    }

    return data;
}

// Fonction pour charger plusieur fichier dans un vecteur
vector<vector<float>> loadFiles(vector<string> filenames) {
    vector<vector<float>> data;

    for (string &filename : filenames) {
        data.push_back(readFile(filename));
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

void loadDataFromFile(vector<vector<float>> &inputs, vector<vector<float>> &desiredOutputs) {
    vector<string> listInputFile = listFiles("data/inputs");
    vector<string> listDesiredOutputFile = listFiles("data/outputs");

    cout << "Inputs  file number : " << listInputFile.size() << endl;
    cout << "Outputs file number : " << listDesiredOutputFile.size() << endl;

    inputs = loadFiles(listInputFile);
    desiredOutputs = loadFiles(listDesiredOutputFile);
}

int main() {
    int numInputs = 2;
    int numOutputs = 2;
    int numHiddenLayers = 5;
    vector<float> hiddenLayer = {3, 2, 4, 7, 5};

    NeuralNetwork nn(numInputs, numHiddenLayers, hiddenLayer, numOutputs);
    
    vector<float> inputValues = {1, 0};
    vector<float> output = nn.feedForward(inputValues);


    vector<vector<float>> inputs;
    vector<vector<float>> desiredOutputs;

    loadDataFromFile(inputs, desiredOutputs);

    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < inputs.size(); i++) {
            output = nn.feedForward(inputs[i]);
            nn.backPropagation(desiredOutputs[i], nn.feedForward(inputs[i]), 0.1);
        }
    }

    output = nn.feedForward(inputValues);

    cout << "Output : " ;

    for (float &value : output) {
        cout << value << " ";
    }
    cout << endl;

    return 0;
}
