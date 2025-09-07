#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

class NeuralNetwork {
private:
    std::vector<int> topology;
    std::vector<std::vector<double>> layers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    double learningRate;
    
    // Activation functions
    double sigmoid(double x);
    double sigmoidDerivative(double x);
    double relu(double x);
    double reluDerivative(double x);
    
    // Random number generator
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

public:
    NeuralNetwork(const std::vector<int>& topology, double lr = 0.01);
    
    // Core functions
    std::vector<double> feedForward(const std::vector<double>& inputs);
    void backPropagate(const std::vector<double>& inputs, 
                      const std::vector<double>& targets);
    void train(const std::vector<std::vector<double>>& inputs,
              const std::vector<std::vector<double>>& targets,
              int epochs);
    
    // Utility functions
    double calculateError(const std::vector<double>& outputs,
                         const std::vector<double>& targets);
    void printWeights();
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
};