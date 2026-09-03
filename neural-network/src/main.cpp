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
    nn.Train(inputs, targets, 10000);
    
    std::cout << "\nTesting trained network:" << std::endl;
    bool all_correct = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = nn.FeedForward(inputs[i]);
        double expected = targets[i][0];
        bool correct = std::abs(output[0] - expected) < 0.1;
        all_correct = all_correct && correct;

        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] ";
        std::cout << "Expected: " << expected << " ";
        std::cout << "Got: " << output[0] << " ";
        std::cout << "[" << (correct ? "PASS" : "FAIL") << "]" << std::endl;
    }
    std::cout << "\n[" << (all_correct ? "PASS" : "FAIL") << "] network learned XOR within tolerance" << std::endl;

    // Round-trip SaveModel()/LoadModel() through a fresh network with the
    // same topology, and confirm it reproduces identical predictions.
    const std::string kModelPath = "xor_model.txt";
    nn.SaveModel(kModelPath);

    NeuralNetwork reloaded(topology, 0.5);
    reloaded.LoadModel(kModelPath);

    bool save_load_matches = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto original_output = nn.FeedForward(inputs[i]);
        auto reloaded_output = reloaded.FeedForward(inputs[i]);
        if (std::abs(original_output[0] - reloaded_output[0]) > 1e-9) {
            save_load_matches = false;
        }
    }
    std::cout << "[" << (save_load_matches ? "PASS" : "FAIL")
               << "] SaveModel()/LoadModel() round-trip reproduces identical predictions" << std::endl;

    return (all_correct && save_load_matches) ? 0 : 1;
}
