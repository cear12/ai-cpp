#include "../neural-network.h"

#include <fstream>
#include <iomanip>
#include <sstream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology, double lr)
    : topology_(topology), learning_rate_(lr), gen_(std::random_device{}()), dis_(-1.0, 1.0) {
    
    // Initialize layers
    layers_.resize(topology.size());
    for (size_t i = 0; i < topology.size(); ++i) {
        layers_[i].resize(topology[i]);
    }
    
    // Initialize weights and biases
    weights_.resize(topology.size() - 1);
    biases_.resize(topology.size() - 1);
    
    for (size_t i = 0; i < topology.size() - 1; ++i) {
        weights_[i].resize(topology[i]);
        biases_[i].resize(topology[i + 1]);
        
        for (int j = 0; j < topology[i]; ++j) {
            weights_[i][j].resize(topology[i + 1]);
            for (int k = 0; k < topology[i + 1]; ++k) {
                weights_[i][j][k] = dis_(gen_);
            }
        }
        
        for (int j = 0; j < topology[i + 1]; ++j) {
            biases_[i][j] = dis_(gen_);
        }
    }
}

double NeuralNetwork::Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::SigmoidDerivative(double x) {
    return x * (1.0 - x);
}

double NeuralNetwork::Relu(double x) {
    return std::max(0.0, x);
}

double NeuralNetwork::ReluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

std::vector<double> NeuralNetwork::FeedForward(const std::vector<double>& inputs) {
    // Set input layer
    layers_[0] = inputs;
    
    // Forward propagation
    for (size_t i = 1; i < topology_.size(); ++i) {
        for (int j = 0; j < topology_[i]; ++j) {
            double sum = biases_[i - 1][j];
            for (int k = 0; k < topology_[i - 1]; ++k) {
                sum += layers_[i - 1][k] * weights_[i - 1][k][j];
            }
            
            // Apply activation function (sigmoid for hidden layers, linear for output)
            if (i == topology_.size() - 1) {
                layers_[i][j] = sum; // Linear activation for output layer
            } else {
                layers_[i][j] = Sigmoid(sum); // Sigmoid for hidden layers
            }
        }
    }
    
    return layers_.back();
}

std::vector<double> NeuralNetwork::BackPropagate(const std::vector<double>& inputs,
                                                  const std::vector<double>& targets) {
    // Forward pass
    FeedForward(inputs);
    
    // Calculate output layer errors
    std::vector<std::vector<double>> errors(topology_.size());
    for (size_t i = 0; i < topology_.size(); ++i) {
        errors[i].resize(topology_[i]);
    }
    
    // Output layer error
    int output_layer = topology_.size() - 1;
    for (int i = 0; i < topology_[output_layer]; ++i) {
        errors[output_layer][i] = targets[i] - layers_[output_layer][i];
    }
    
    // Hidden layers error (backpropagate)
    for (int i = output_layer - 1; i >= 1; --i) {
        for (int j = 0; j < topology_[i]; ++j) {
            double error = 0.0;
            for (int k = 0; k < topology_[i + 1]; ++k) {
                error += errors[i + 1][k] * weights_[i][j][k];
            }
            errors[i][j] = error * SigmoidDerivative(layers_[i][j]);
        }
    }
    
    // Update weights and biases
    for (size_t i = 0; i < weights_.size(); ++i) {
        for (int j = 0; j < topology_[i]; ++j) {
            for (int k = 0; k < topology_[i + 1]; ++k) {
                weights_[i][j][k] += learning_rate_ * errors[i + 1][k] * layers_[i][j];
            }
        }
        
        for (int j = 0; j < topology_[i + 1]; ++j) {
            biases_[i][j] += learning_rate_ * errors[i + 1][j];
        }
    }

    // FeedForward(inputs) at the top of this function already computed and
    // stored the network's output in layers_.back(); returning it here lets
    // Train() reuse it instead of redundantly running a second forward pass
    // per sample per epoch just to compute the training error.
    return layers_.back();
}

void NeuralNetwork::Train(const std::vector<std::vector<double>>& inputs,
                         const std::vector<std::vector<double>>& targets,
                         int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = BackPropagate(inputs[i], targets[i]);
            total_error += CalculateError(output, targets[i]);
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Error: " << total_error / inputs.size() << std::endl;
        }
    }
}

double NeuralNetwork::CalculateError(const std::vector<double>& outputs,
                                   const std::vector<double>& targets) {
    double error = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        double diff = targets[i] - outputs[i];
        error += diff * diff;
    }
    return error * 0.5;
}

void NeuralNetwork::PrintWeights() {
    for (size_t i = 0; i < weights_.size(); ++i) {
        std::cout << "Layer " << i << " -> " << i + 1 << " weights:" << std::endl;
        for (size_t j = 0; j < weights_[i].size(); ++j) {
            for (size_t k = 0; k < weights_[i][j].size(); ++k) {
                std::cout << weights_[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void NeuralNetwork::SaveModel(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: could not open " << filename << " for writing" << std::endl;
        return;
    }

    // 17 significant decimal digits is the number required to round-trip
    // any IEEE-754 double exactly (std::numeric_limits<double>::max_digits10).
    // Without this, operator<<'s default 6-digit precision silently
    // truncates every saved weight -- the model still "loads", it's just
    // quietly a different (nearby, but not identical) network.
    out << std::setprecision(17);

    // Simple whitespace-separated text format: topology, then learning
    // rate, then every weight and bias in the same nested order the
    // constructor initializes them in.
    out << topology_.size() << "\n";
    for (int size : topology_) {
        out << size << " ";
    }
    out << "\n" << learning_rate_ << "\n";

    for (const auto& layer_weights : weights_) {
        for (const auto& neuron_weights : layer_weights) {
            for (double w : neuron_weights) {
                out << w << " ";
            }
        }
        out << "\n";
    }

    for (const auto& layer_biases : biases_) {
        for (double b : layer_biases) {
            out << b << " ";
        }
        out << "\n";
    }
}

void NeuralNetwork::LoadModel(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Error: could not open " << filename << " for reading" << std::endl;
        return;
    }

    std::size_t topology_size = 0;
    in >> topology_size;

    std::vector<int> loaded_topology(topology_size);
    for (auto& size : loaded_topology) {
        in >> size;
    }

    if (loaded_topology != topology_) {
        std::cerr << "Error: " << filename << " topology does not match this network's topology"
                   << std::endl;
        return;
    }

    in >> learning_rate_;

    for (auto& layer_weights : weights_) {
        for (auto& neuron_weights : layer_weights) {
            for (double& w : neuron_weights) {
                in >> w;
            }
        }
    }

    for (auto& layer_biases : biases_) {
        for (double& b : layer_biases) {
            in >> b;
        }
    }

    if (!in) {
        std::cerr << "Error: " << filename << " was truncated or malformed" << std::endl;
    }
}
