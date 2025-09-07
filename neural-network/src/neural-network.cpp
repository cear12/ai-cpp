#include "../neural-network.h"

#include <fstream>
#include <sstream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology, double lr)
    : topology(topology), learningRate(lr), gen(std::random_device{}()), dis(-1.0, 1.0) {
    
    // Initialize layers
    layers.resize(topology.size());
    for (size_t i = 0; i < topology.size(); ++i) {
        layers[i].resize(topology[i]);
    }
    
    // Initialize weights and biases
    weights.resize(topology.size() - 1);
    biases.resize(topology.size() - 1);
    
    for (size_t i = 0; i < topology.size() - 1; ++i) {
        weights[i].resize(topology[i]);
        biases[i].resize(topology[i + 1]);
        
        for (int j = 0; j < topology[i]; ++j) {
            weights[i][j].resize(topology[i + 1]);
            for (int k = 0; k < topology[i + 1]; ++k) {
                weights[i][j][k] = dis(gen);
            }
        }
        
        for (int j = 0; j < topology[i + 1]; ++j) {
            biases[i][j] = dis(gen);
        }
    }
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

double NeuralNetwork::relu(double x) {
    return std::max(0.0, x);
}

double NeuralNetwork::reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& inputs) {
    // Set input layer
    layers[0] = inputs;
    
    // Forward propagation
    for (size_t i = 1; i < topology.size(); ++i) {
        for (int j = 0; j < topology[i]; ++j) {
            double sum = biases[i - 1][j];
            for (int k = 0; k < topology[i - 1]; ++k) {
                sum += layers[i - 1][k] * weights[i - 1][k][j];
            }
            
            // Apply activation function (sigmoid for hidden layers, linear for output)
            if (i == topology.size() - 1) {
                layers[i][j] = sum; // Linear activation for output layer
            } else {
                layers[i][j] = sigmoid(sum); // Sigmoid for hidden layers
            }
        }
    }
    
    return layers.back();
}

void NeuralNetwork::backPropagate(const std::vector<double>& inputs,
                                 const std::vector<double>& targets) {
    // Forward pass
    feedForward(inputs);
    
    // Calculate output layer errors
    std::vector<std::vector<double>> errors(topology.size());
    for (size_t i = 0; i < topology.size(); ++i) {
        errors[i].resize(topology[i]);
    }
    
    // Output layer error
    int outputLayer = topology.size() - 1;
    for (int i = 0; i < topology[outputLayer]; ++i) {
        errors[outputLayer][i] = targets[i] - layers[outputLayer][i];
    }
    
    // Hidden layers error (backpropagate)
    for (int i = outputLayer - 1; i >= 1; --i) {
        for (int j = 0; j < topology[i]; ++j) {
            double error = 0.0;
            for (int k = 0; k < topology[i + 1]; ++k) {
                error += errors[i + 1][k] * weights[i][j][k];
            }
            errors[i][j] = error * sigmoidDerivative(layers[i][j]);
        }
    }
    
    // Update weights and biases
    for (size_t i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < topology[i]; ++j) {
            for (int k = 0; k < topology[i + 1]; ++k) {
                weights[i][j][k] += learningRate * errors[i + 1][k] * layers[i][j];
            }
        }
        
        for (int j = 0; j < topology[i + 1]; ++j) {
            biases[i][j] += learningRate * errors[i + 1][j];
        }
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                         const std::vector<std::vector<double>>& targets,
                         int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            backPropagate(inputs[i], targets[i]);
            auto output = feedForward(inputs[i]);
            totalError += calculateError(output, targets[i]);
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Error: " << totalError / inputs.size() << std::endl;
        }
    }
}

double NeuralNetwork::calculateError(const std::vector<double>& outputs,
                                   const std::vector<double>& targets) {
    double error = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        double diff = targets[i] - outputs[i];
        error += diff * diff;
    }
    return error * 0.5;
}

void NeuralNetwork::printWeights() {
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << "Layer " << i << " -> " << i + 1 << " weights:" << std::endl;
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                std::cout << weights[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
