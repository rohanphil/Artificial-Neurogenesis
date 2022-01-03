#pragma once

#include <vector>
#include <string>
#include <iostream>
//#include "source.cpp"

using namespace std;

class ANG {
	vector<vector<float>> activations;
	vector<vector<vector<float>>> derivatives;
	vector<vector<vector<float>>> weights;
	vector<vector<vector<float>>> seed_weights, seed_derivatives;
	vector<vector<float>> seed_activations;
	
public:
	//Default Constructor
	ANG();
	//Parameterized Constructor
	ANG(vector<int> hidden_layers, int num_outputs = 1, int num_inputs = 10);

	//Sort Index
	vector<int> sorted_index(int size, vector<float> b);

	//Print a 3D vector (weights, derivatives)
	void Print3D(vector<vector<vector<float>>> a);

	//Extract Standard Deviation
	float stdev(vector<float> data);

	//Print 2D Vector
	void PrintMatrix(vector<vector<float>> a);

	//Print a 1D vector
	void PrintVector(vector<float> a);

	//Transpose a 2D vector
	vector<vector<float>> transpose(vector<vector<float>> b);

	//Not in use any longer
	void Init(vector<int> hidden_layers, int num_outputs = 1, int num_inputs = 10);

	//Dot Product calculation
	vector<vector<float>> dot_new(vector<vector<float>>& a, vector<vector<float>>& b);

	//Deprecated dot product (i think) 
	vector<vector<float>> dot_product(vector<vector<float>>& a, vector<vector<float>>& b);

	//Calculate mean squared error
	float _mse(vector<float> target, vector<float> output);

	//Sigmoid activation
	vector<float> _sigmoid(vector<float>& a);

	//Sigmoid derivative
	vector<float> _sigmoid_derivative(vector<float>& a);

	//Forward propogation
	vector<float> forward_propogate(vector<float> inputs);

	//Simple test copy of FP might remove in the future
	vector<float> forward_propogate_test(vector<float> inputs);

	//Gradient descent
	void gradient_descent(float learning_rate = 1.0);

	//Back propogation
	void back_propogate(vector<float> error);

	//Training 
	void train(vector<vector<float>> inputs, vector<vector<float>> targets, int epochs, float learning_rate);

	//Seed network creation
	void create_seed();

	//Priming base Network;
	void prime_base_network(vector<vector<float>> inputs, vector<vector<float>> targets, int cycles, float learning_rate = 0.5);

	//remove temp classifier
	void remove_temp_classifier();

	//Add destination layer
	void add_destination_layer();

	//Add classification layer
	void add_class_layer(vector<vector<float>> targets);

	//Add final classification layer
	void add_class_layer_final();

	//Create a temp classifier for the seed network
	void create_temp_classifier_seed(vector<vector<float>> targets);

	//Operations on the seed network

	vector<float> forward_propogate_seed(vector<float> inputs);

	void back_propogate_seed(vector<float> error);

	void train_seed(vector<vector<float>> inputs, vector<vector<float>> targets, int epochs, float learning_rate);

	void prime_seed_network(vector<vector<float>> inputs, vector<vector<float>> targets, int cycles, float learning_rate = 0.5);

	void remove_temp_classifier_seed();

	//Extract extreme member classes
	vector<vector<vector<float>>> extreme_member_classes(vector<vector<float>> inputs, vector<vector<float>> targets);

	vector<float> return_source_activations();
	
	float return_acc(vector<vector<float>> inputs, vector<vector<float>> targets);

	//Grow the network
	void ANG_grow(vector<vector<float>> inputs, vector<vector<float>> targets);

	//Save/Load the model
	void save_weights_der(vector<vector<vector<float>>> Matrix, string name);

	void save_activations(vector<vector<float>> Matrix, string name);

	void read_weights_der(vector<vector<vector<float>>>& Matrix, string name);

	void read_activations(vector<vector<float>>& Act, string name);

	void save_model(string name);

	void load_model(string name);
};