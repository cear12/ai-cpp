#include "../neural-network.h"

#include <iostream>
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
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = nn.feedForward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] ";
        std::cout << "Expected: " << targets[i][0] << " ";
        std::cout << "Got: " << output[0] << std::endl;
    }
    
    return 0;
}
