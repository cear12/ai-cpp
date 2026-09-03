#include "../neural-network.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // Create neural network topology: 2 inputs, 4 hidden neurons, 1 output
    std::vector<int> topology = {2, 4, 1};
    NeuralNetwork nn(topology, 0.5);
    
    // XOR training data
    std::vector<std::vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    
    std::vector<std::vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };
    
    std::cout << "Training XOR Neural Network..." << std::endl;
    nn.train(inputs, targets, 10000);
    
    std::cout << "\nTesting trained network:" << std::endl;
    bool allCorrect = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = nn.feedForward(inputs[i]);
        double expected = targets[i][0];
        bool correct = std::abs(output[0] - expected) < 0.1;
        allCorrect = allCorrect && correct;

        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] ";
        std::cout << "Expected: " << expected << " ";
        std::cout << "Got: " << output[0] << " ";
        std::cout << "[" << (correct ? "PASS" : "FAIL") << "]" << std::endl;
    }
    std::cout << "\n[" << (allCorrect ? "PASS" : "FAIL") << "] network learned XOR within tolerance" << std::endl;

    // Round-trip saveModel()/loadModel() through a fresh network with the
    // same topology, and confirm it reproduces identical predictions.
    const std::string modelPath = "xor_model.txt";
    nn.saveModel(modelPath);

    NeuralNetwork reloaded(topology, 0.5);
    reloaded.loadModel(modelPath);

    bool saveLoadMatches = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto originalOutput = nn.feedForward(inputs[i]);
        auto reloadedOutput = reloaded.feedForward(inputs[i]);
        if (std::abs(originalOutput[0] - reloadedOutput[0]) > 1e-9) {
            saveLoadMatches = false;
        }
    }
    std::cout << "[" << (saveLoadMatches ? "PASS" : "FAIL")
               << "] saveModel()/loadModel() round-trip reproduces identical predictions" << std::endl;

    return (allCorrect && saveLoadMatches) ? 0 : 1;
}
