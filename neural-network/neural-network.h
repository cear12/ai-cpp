#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <string>

class NeuralNetwork {
private:
    std::vector<int> topology_;
    std::vector<std::vector<double>> layers_;
    std::vector<std::vector<std::vector<double>>> weights_;
    std::vector<std::vector<double>> biases_;
    double learning_rate_;
    
    // Activation functions
    double Sigmoid(double x);
    double SigmoidDerivative(double x);
    double Relu(double x);
    double ReluDerivative(double x);
    
    // Random number generator
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dis_;

public:
    NeuralNetwork(const std::vector<int>& topology, double lr = 0.01);
    
    // Core functions
    std::vector<double> FeedForward(const std::vector<double>& inputs);
    std::vector<double> BackPropagate(const std::vector<double>& inputs,
                                       const std::vector<double>& targets);
    void Train(const std::vector<std::vector<double>>& inputs,
              const std::vector<std::vector<double>>& targets,
              int epochs);
    
    // Utility functions
    double CalculateError(const std::vector<double>& outputs,
                         const std::vector<double>& targets);
    void PrintWeights();
    void SaveModel(const std::string& filename);
    void LoadModel(const std::string& filename);
};