#include <iostream>
#include <time.h>
#include <vector>
#include <typeinfo>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/serialization/vector.hpp>
//#include <pybind11/pybind11.h>
//#include <Python.h>
//#include <pybind11/embed.h>

using namespace std;

class ANG
{

	vector<vector<float>> activations;
	vector<vector<vector<float>>> derivatives;
	vector<vector<vector<float>>> weights;
	vector<vector<vector<float>>> seed_weights, seed_derivatives;
	vector<vector<float>> seed_activations;
	/*
private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& weights;
		ar& derivatives;
		ar& activations;
		ar& seed_weights;
		ar& seed_derivatives;
		ar& seed_activations;
	}
	*/

public:
	//Default Constructor
	ANG(){
		activations = {};
		derivatives = {};
		weights = {};
		seed_weights = {};
		seed_derivatives = {};
		seed_activations = {};
	}
	//Parameterized Constructor
	ANG(vector<int> hidden_layers, int num_outputs = 1, int num_inputs = 10) {
		vector<int> layers;
		layers.push_back(num_inputs);
		for (int i = 0; i < hidden_layers.size(); i++) layers.push_back(hidden_layers[i]);
		layers.push_back(num_outputs);
		//for (int lay = 0; lay < layers.size(); lay++) cout << layers[lay] << endl;
		vector<vector<float>> w;
		for (int i = 0; i < layers.size() - 1; i++) {
			int row_len, col_len;
			row_len = layers[i];
			col_len = layers[i + 1];
			//srand(time(0));
			vector < vector <float>> Matrix(row_len, vector<float>(col_len, 0));
			for (auto it_row = Matrix.begin(); it_row != Matrix.end(); it_row++)
			{
				// Getting each (i,j) element and assigning random value to it
				for (auto it_col = it_row->begin(); it_col != it_row->end(); it_col++)
				{
					*it_col = (float)rand() / RAND_MAX;
					//cout << *it_col;
				}
			}
			w = Matrix;
			weights.push_back(w);
		}
		//for (int i = 0; i < w.size(); i++) {
			//for (int j = 0; j < w[i].size(); j++)
				//cout << w[i][j] << " ";  
			//cout << endl;
		//}

		vector<vector<float>> d;
		for (int i = 0; i < layers.size() - 1; i++) {
			int row_len, col_len;
			row_len = layers[i];
			col_len = layers[i + 1];
			vector < vector <float>> Matrix(row_len, vector<float>(col_len, 0.0));
			d = Matrix;
			derivatives.push_back(d);
		}
		//for (int i = 0; i < w.size(); i++) {
			//for (int j = 0; j < w[i].size(); j++)
				//cout << w[i][j] << " ";  
			//cout << endl;
		//}


		vector<float> a;
		for (int i = 0; i < layers.size(); i++) {
			int row_len, col_len;
			row_len = layers[i];
			vector<float> Matrix(row_len, 0);
			a = Matrix;
			activations.push_back(a);
		}
		//for (int i = 0; i < w.size(); i++) {
			//for (int j = 0; j < w[i].size(); j++)
				//cout << w[i][j] << " ";  
			//cout << endl;
		//}
		//cout << weights.size() << " ";
		//for (int i = 0; i < weights.size(); i++) {
			//for (int j = 0; j < weights[i].size(); j++) {
			//	for (int k = 0; k < weights[i][j].size(); k++)
			//		cout << weights[i][j][k] << " ";
			//	cout << endl;
			//}
		//}
	}
	//friend class boost::serialization::access;
	vector<int> sorted_index(int size, vector<float> b) {
		vector<int> indices(size);
		std::iota(indices.begin(), indices.end(), 0);
		sort(indices.begin(), indices.end(), [&](int A, int B) -> bool {return b[A] < b[B];});
		return indices;
	}

	void Print3D(vector<vector<vector<float>>> a) {
		for (int i = 0; i < a.size(); i++) {
			for (int j = 0; j < a[i].size(); j++) {
				for (int k = 0; k < a[i][j].size(); k++) {
					cout << a[i][j][k] << " ";
				}
				cout << endl;
			}
			cout << "#" << endl;
		}
	}
	float stdev(vector<float> data) {
		int size = data.size();
		float mean, sd = 0.0, sum = 0.0;
		for (int i = 0; i < size; i++) sum += data[i];
		mean = (float)sum / size;
		for (int i = 0; i < size; i++) sd += pow(data[i] - mean, 2);
		return sqrt(sd / size);

	}
	void PrintMatrix(vector<vector<float>> a) {
		for (int i = 0; i < a.size(); i++) {

			for (int j = 0; j < a[i].size(); j++) cout << a[i][j] << " ";
			cout << endl;
		}
	}

	void PrintVector(vector<float> a) {
		for (int i = 0; i < a.size(); i++) cout << a[i] << " ";
	}

	vector<vector<float>> transpose(vector<vector<float>> b)
	{
		if (b.size() == 0)
			return { {} };
		vector<vector<float>> trans_vec(b[0].size(), vector<float>());
		for (int i = 0; i < b.size(); i++)
		{
			for (int j = 0; j < b[i].size(); j++)
			{
				trans_vec[j].push_back(b[i][j]);
			}
		}
		return trans_vec;   
	}
	//init not in use anymore
	void Init(vector<int> hidden_layers, int num_outputs = 1, int num_inputs = 10)
	{
		vector<int> layers;
		layers.push_back(num_inputs);
		for (int i = 0; i < hidden_layers.size(); i++) layers.push_back(hidden_layers[i]);
		layers.push_back(num_outputs);
		//for (int lay = 0; lay < layers.size(); lay++) cout << layers[lay] << endl;
		vector<vector<float>> w;
		for (int i = 0; i < layers.size() - 1; i++) {
			int row_len, col_len;
			row_len = layers[i];
			col_len = layers[i + 1];
			//srand(time(0));
			vector < vector <float>> Matrix(row_len, vector<float>(col_len, 0));
			for (auto it_row = Matrix.begin(); it_row != Matrix.end(); it_row++)
			{
				// Getting each (i,j) element and assigning random value to it
				for (auto it_col = it_row->begin(); it_col != it_row->end(); it_col++)
				{
					*it_col = (float)rand()/RAND_MAX;
					//cout << *it_col;
				}
			}
			w = Matrix;
			weights.push_back(w);
		}
		//for (int i = 0; i < w.size(); i++) {
			//for (int j = 0; j < w[i].size(); j++)
				//cout << w[i][j] << " ";  
			//cout << endl;
		//}
		
		vector<vector<float>> d;
		for (int i = 0; i < layers.size() - 1; i++) {
			int row_len, col_len;
			row_len = layers[i];
			col_len = layers[i + 1];
			vector < vector <float>> Matrix(row_len, vector<float>(col_len, 0.0));
			d = Matrix;
			derivatives.push_back(d);
		}
		//for (int i = 0; i < w.size(); i++) {
			//for (int j = 0; j < w[i].size(); j++)
				//cout << w[i][j] << " ";  
			//cout << endl;
		//}
		
		
		vector<float> a;
		for (int i = 0; i < layers.size(); i++) {
			int row_len, col_len;
			row_len = layers[i];
			vector<float> Matrix(row_len, 0);
			a = Matrix;
			activations.push_back(a);
		}
		//for (int i = 0; i < w.size(); i++) {
			//for (int j = 0; j < w[i].size(); j++)
				//cout << w[i][j] << " ";  
			//cout << endl;
		//}
		//cout << weights.size() << " ";
		//for (int i = 0; i < weights.size(); i++) {
			//for (int j = 0; j < weights[i].size(); j++) {
			//	for (int k = 0; k < weights[i][j].size(); k++)
			//		cout << weights[i][j][k] << " ";
			//	cout << endl;
			//}
		//}
	}/**//**/
	vector<vector<float>> dot_new(vector<vector<float>>& a, vector<vector<float>>& b) {
		int row_a, row_b, col_a, col_b;
		row_a = a.size();
		row_b = b.size();
		col_a = a[0].size();
		col_b = b[0].size();
		vector<vector<float>> dot;
		for (int i = 0; i < row_a; i++) dot.push_back({});
		for (int i = 0; i < row_a; i++) {
			vector<float> col0;
			for (int j = 0; j < col_b; j++) col0.push_back(0);
			dot[i] = col0;
		}
		//cout << row_a << " " << col_a << " " << row_b << " " << col_b << endl;
		for (int i = 0; i < row_a; i++) {
			for (int j = 0; j < col_b; j++) {
				dot[i][j] = 0;
				for (int k = 0; k < col_a; k++)
					dot[i][j] = dot[i][j] + (a[i][k] * b[k][j]);
			}
		}
		return dot;

	}
	//Dot product done
	vector<vector<float>> dot_product(vector<vector<float>> &a, vector<vector<float>> &b) {
		int row_a, row_b, col_a, col_b;
		row_a = a.size();
		row_b = b.size();
		col_a = a[0].size();
		col_b = b[0].size();
		vector<vector<float>> dot;
		dot.push_back({});
		for (int i = 0; i < col_b; i++)
			dot[0].push_back(0);
		for (int i = 0; i < row_a; i++) {
			for (int j = 0; j < col_b; j++) {
				dot[i][j] = 0;
				for(int k = 0; k< col_a; k++)
					dot[i][j] = dot[i][j] + (a[i][k] * b[k][j]);
			}
		}

		return dot;


	}

	float _mse(vector<float> target, vector<float> output) {
		float sum = 0;
		int n = target.size();
		for (int i = 0; i < target.size(); i++) {
			sum = sum + ((target[i] - output[i]) * (target[i] - output[i]));
		}
		return sum / n;
	}
	//sigmoid done
	vector<float> _sigmoid(vector<float>& a) {
		vector<float> sigmoid;
		for (int i = 0; i < a.size(); i++) sigmoid.push_back(1.0 / (1.0 + (float)exp(-1 * a[i])));
		return sigmoid;
	}

	// sigmoid derivative done
	vector<float> _sigmoid_derivative(vector<float>& a) {
		vector<float> sigmoid;
		for (int i = 0; i < a.size(); i++) sigmoid.push_back(a[i] * (1 - a[i]));
		return sigmoid;
	}
	//forward propogation done
	vector<float> forward_propogate(vector<float> inputs) {
		activations[0] = inputs;
		vector<vector<float>> net_inputs;
		vector<vector<float>> activ;
		activ.push_back(inputs);
		for (int i = 0; i < weights.size(); i++) {
			//cout << weights[i].size() << " " << endl;
			net_inputs = dot_new(activ, weights[i]);
			vector<float> temp_activ;
			temp_activ = _sigmoid(net_inputs[0]);
			activ.pop_back();
			activ.push_back(temp_activ);
			activations[i + 1] = temp_activ;
		}
		//return activations and change return type of function
		return activ[0];
	}

	vector<float> forward_propogate_test(vector<float> inputs) {
		activations[0] = inputs;
		vector<vector<float>> net_inputs;
		vector<vector<float>> activ;
		activ.push_back(inputs);
		for (int i = 0; i < weights.size(); i++) {
			//cout << weights[i].size() << " " << endl;
			net_inputs = dot_new(activ, weights[i]);
			vector<float> temp_activ;
			temp_activ = _sigmoid(net_inputs[0]);
			activ.pop_back();
			activ.push_back(temp_activ);
			activations[i + 1] = temp_activ;
		}
		//return activations and change return type of function
		return activ[0];
	}

	void gradient_descent(float learning_rate = 1.0) {
		for (int i = 0; i < weights.size(); i++) {
			//vector<vector<float>> w = weights[i];
			//weights[i] += weights[i] * learning_rate;
			for (int j = 0; j < weights[i].size(); j++) {
				//cout << weights[i][j].size();
				//transform(weights[i][j].begin(), weights[i][j].end(), temp.begin(), [learning_rate](float& c) {return c * learning_rate; });
				for (int k = 0; k < weights[i][j].size(); k++) {
					weights[i][j][k] = weights[i][j][k] + derivatives[i][j][k] * learning_rate;
				}
				//transform(weights[i][j].begin(), weights[i][j].end(), temp.begin(), weights[i][j].begin(), plus<float>());
			}
		}
	}

	void back_propogate(vector<float> error) {
		vector<float> activation;
		//vector<float> delta;
		vector<float> sigmoid_derivative;
		//cout << activations.size();
		for (int i = derivatives.size()-1; i >= 0; i--) {
			//cout << i;
			vector<float> delta;
			activation = activations[i + 1];
			//cout << activation.size() << endl;
			sigmoid_derivative = _sigmoid_derivative(activation);
			//cout << sigmoid_derivative.size() <<endl;
			//PrintVector(error);
			for (int j = 0; j < sigmoid_derivative.size(); j++) {
				delta.push_back(sigmoid_derivative[j] * error[j]);
			}
			//cout << delta.size();
			vector<vector<float>> delta_re;
			delta_re.push_back(delta);
			vector<float> current_acc = activations[i];
			vector<vector<float>> current_activations;
			for (int k = 0; k < current_acc.size(); k++) current_activations.push_back({ current_acc[k] });
			//PrintMatrix(current_activations);
			//delta = std::transform(sigmoid_derivative.begin(), sigmoid_derivative.end(), error, std::multiplies<float>());
			//#cout << delta_re[0].size()<< endl;
			vector<vector<float>> dotp = dot_new(current_activations, delta_re);
			derivatives[i] = dotp;
			vector<vector<float>> t_weights = transpose(weights[i]);
			vector<vector<float>> delta_mul = { delta };
			error = dot_new(delta_mul, t_weights)[0];
			//PrintVector(error);
		}
	}

	void train(vector<vector<float>> inputs, vector<vector<float>> targets, int epochs, float learning_rate) {
		for (int i = 0; i < epochs; i++) {
			float sum_errors = 0;
			for (int j = 0; j < inputs.size(); j++) {
				vector<float> target = targets[j];
				vector<float> output = forward_propogate(inputs[j]);
				vector<float> error;
				for (int k = 0; k < output.size(); k++)
					error.push_back(target[k] - output[k]);
				back_propogate(error);
				gradient_descent(learning_rate);
				sum_errors = sum_errors + _mse(target, output);
			}
			cout << "Error: " << sum_errors / inputs.size() << " at epoch" << i + 1 << endl;
		}
		cout << "Training complete" << endl;
		cout << "=======" << endl;
	}

	void create_seed() {
		seed_weights = weights;
		seed_activations = activations;
		//cout << activations.size() << endl;
		seed_derivatives = derivatives;
		seed_weights.resize(seed_weights.size() - 2);
		seed_activations.resize(seed_activations.size() - 2);
		//cout << seed_activations.size() << endl;
		seed_derivatives.resize(seed_derivatives.size() - 2);
	}

	void prime_base_network(vector<vector<float>> inputs, vector<vector<float>> targets, int cycles, float learning_rate = 0.5) {
		train(inputs, targets, cycles, learning_rate);
		Print3D(weights);
	}

	void remove_temp_classifier() {
		weights.resize(weights.size() - 1);
		derivatives.resize(derivatives.size() - 1);
		activations.resize(activations.size() - 1);
	}

	void add_destination_layer() {
		vector<vector<float>> temp;
		int r, c;
		// add new weights 
		temp = weights.back();
		//PrintMatrix(temp);
		r = temp[0].size(); //changed to set the size of the destionation layer to a minimum 3
		c = 1;// temp[0].size();
		vector < vector <float>> Matrix(r, vector<float>(c, 0));
		for (auto it_row = Matrix.begin(); it_row != Matrix.end(); it_row++)
		{
			// Getting each (i,j) element and assigning random value to it
			for (auto it_col = it_row->begin(); it_col != it_row->end(); it_col++)
			{
				*it_col = (float)rand() / RAND_MAX;
				//cout << *it_col;
			}
		}
		temp = Matrix;
		weights.push_back(temp);
		//for (int wei = 0; wei < weights.size(); wei++) cout << "The matrices are of shape " << weights[wei].size() << " x " << weights[wei][0].size() << endl;
		// add new derivatives
		temp = derivatives.back();
		vector < vector <float>> Matrix_dr(r, vector<float>(c, 0.0));
		temp = Matrix_dr;
		derivatives.push_back(temp);

		// add new activations
		vector<float> temp_acc = activations.back();
		int row_len = 1;// temp_acc.size();
		vector<float> Matrix_acc(row_len, 0);
		temp_acc = Matrix_acc;
		activations.push_back(temp_acc);
		//for (int wei = 0; wei < activations.size(); wei++) cout << "The matrices are of shape " << activations.size() << " x " << activations[wei].size() << endl;
	}
    //done
	void add_class_layer(vector<vector<float>> targets) {
		vector<float> temp;
		for (int i = 0; i < targets.size(); i++) temp.push_back(targets[i][0]);
		sort(temp.begin(), temp.end());
		int num_classes = std::unique(temp.begin(), temp.end()) - temp.begin();
		num_classes = num_classes - 1;
		vector<vector<float>> temp_w;
		int r, c;
		temp_w = weights.back();
		r = temp_w[0].size();
		c = num_classes;
		vector < vector <float>> Matrix(r, vector<float>(c, 0));
		for (auto it_row = Matrix.begin(); it_row != Matrix.end(); it_row++)
		{
			// Getting each (i,j) element and assigning random value to it
			for (auto it_col = it_row->begin(); it_col != it_row->end(); it_col++)
			{
				*it_col = (float)rand() / RAND_MAX;
				//cout << *it_col;
			}
		}
		temp_w = Matrix;
		weights.push_back(temp_w);

		vector<vector<float>> temp_d;
		temp_d = derivatives.back();
		vector < vector <float>> Matrix_dr(r, vector<float>(c, 0.0));
		temp_d = Matrix_dr;
		derivatives.push_back(temp_d);

		vector<float> temp_acc = activations.back();
		int row_len = num_classes;
		vector<float> Matrix_acc(row_len, 0);
		temp_acc = Matrix_acc;
		activations.push_back(temp_acc);


	}
	// not done
	void add_class_layer_final() {}

	void create_temp_classifier_seed(vector<vector<float>> targets) {
		vector<float> temp;
		//cout << seed_weights.size() << endl;
		for (int i = 0; i < targets.size(); i++) temp.push_back(targets[i][0]);
		sort(temp.begin(), temp.end());
		int num_classes = std::unique(temp.begin(), temp.end()) - temp.begin();
		num_classes = num_classes - 1;
		vector<vector<float>> temp_w;
		int r, c;
		temp_w = seed_weights.back();
		r = temp_w[0].size();
		c = num_classes;
		vector < vector <float>> Matrix(r, vector<float>(c, 0));
		//PrintMatrix(Matrix);
		for (auto it_row = Matrix.begin(); it_row != Matrix.end(); it_row++)
		{
			// Getting each (i,j) element and assigning random value to it
			for (auto it_col = it_row->begin(); it_col != it_row->end(); it_col++)
			{
				*it_col = (float)rand() / RAND_MAX;
				//cout << *it_col;
			}
		}
		temp_w = Matrix;
		//PrintMatrix(temp_w);
		seed_weights.push_back(temp_w);
		//cout << seed_weights.size()<< endl;
		vector<vector<float>> temp_d;
		temp_d = seed_derivatives.back();
		vector < vector <float>> Matrix_dr(r, vector<float>(c, 0.0));
		temp_d = Matrix_dr;
		seed_derivatives.push_back(temp_d);

		vector<float> temp_acc = seed_activations.back();
		int row_len = num_classes;
		vector<float> Matrix_acc(row_len, 0);
		temp_acc = Matrix_acc;
		seed_activations.push_back(temp_acc);


	}

	vector<float> forward_propogate_seed(vector<float> inputs) {
		seed_activations[0] = inputs;
		vector<vector<float>> net_inputs;
		vector<vector<float>> activ;
		activ.push_back(inputs);
		for (int i = 0; i < seed_weights.size(); i++) {
			//cout << weights[i].size() << " " << endl;
			net_inputs = dot_new(activ, seed_weights[i]);
			vector<float> temp_activ;
			temp_activ = _sigmoid(net_inputs[0]);
			activ.pop_back();
			activ.push_back(temp_activ);
			seed_activations[i + 1] = temp_activ;
		}
		//return activations and change return type of function
		return activ[0];
	}
	
	void back_propogate_seed(vector<float> error) {
		vector<float> activation;
		//vector<float> delta;
		vector<float> sigmoid_derivative;
		//cout << activations.size();
		for (int i = seed_derivatives.size() - 1; i >= 0; i--) {
			//cout << i;
			vector<float> delta;
			activation = seed_activations[i + 1];
			//cout << activation.size() << endl;
			sigmoid_derivative = _sigmoid_derivative(activation);
			//cout << sigmoid_derivative.size() <<endl;
			//PrintVector(error);
			for (int j = 0; j < sigmoid_derivative.size(); j++) {
				delta.push_back(sigmoid_derivative[j] * error[j]);
			}
			//cout << delta.size();
			vector<vector<float>> delta_re;
			delta_re.push_back(delta);
			vector<float> current_acc = seed_activations[i];
			vector<vector<float>> current_activations;
			for (int k = 0; k < current_acc.size(); k++) current_activations.push_back({ current_acc[k] });
			//PrintMatrix(current_activations);
			//delta = std::transform(sigmoid_derivative.begin(), sigmoid_derivative.end(), error, std::multiplies<float>());
			//#cout << delta_re[0].size()<< endl;
			vector<vector<float>> dotp = dot_new(current_activations, delta_re);
			seed_derivatives[i] = dotp;
			vector<vector<float>> t_weights = transpose(weights[i]);
			vector<vector<float>> delta_mul = { delta };
			error = dot_new(delta_mul, t_weights)[0];
			//PrintVector(error);
		}
	}

	void train_seed(vector<vector<float>> inputs, vector<vector<float>> targets, int epochs, float learning_rate) {
		for (int i = 0; i < epochs; i++) {
			float sum_errors = 0;
			for (int j = 0; j < inputs.size(); j++) {
				vector<float> target = targets[j];
				vector<float> output = forward_propogate_seed(inputs[j]);
				vector<float> error;
				//PrintVector(output);
				for (int k = 0; k < target.size(); k++)
					error.push_back(target[k] - output[k]);
				back_propogate_seed(error);
				gradient_descent(learning_rate);
				sum_errors = sum_errors + _mse(target, output);
			}
			//cout << "Error: " << sum_errors / inputs.size() << " at epoch" << i + 1 << endl;
		}
		cout << "Training complete" << endl;
		cout << "=======" << endl;
	}

	void prime_seed_network(vector<vector<float>> inputs, vector<vector<float>> targets, int cycles, float learning_rate = 0.5) {
		train_seed(inputs, targets, cycles, learning_rate);
	}

	void remove_temp_classifier_seed() {
		seed_weights.resize(seed_weights.size() - 1);
		seed_derivatives.resize(seed_derivatives.size() - 1);
		seed_activations.resize(seed_activations.size() - 1);
	}

	vector<vector<vector<float>>> extreme_member_classes(vector<vector<float>> inputs, vector<vector<float>> targets) {
		//cout << weights.size() << endl;
		create_seed();
		create_temp_classifier_seed(targets);
		prime_seed_network(inputs, targets, 10);
		remove_temp_classifier_seed();
		//cout << weights.size() << endl;
		//cout << weights.size() << endl;
		vector<float> classes, temp;
		for (int i = 0; i < targets.size(); i++) classes.push_back(targets[i][0]);
		temp = classes;
		sort(classes.begin(), classes.end());
		vector<float>::iterator ip = std::unique(classes.begin(), classes.end());
		classes.resize(std::distance(classes.begin(), ip));
		//PrintVector(classes);
		vector<vector<vector<float>>> sorted_classes_list;
		for (int i = 0; i < classes.size(); i++) {
			int count = 0;
			for (int j = 0; j < targets.size(); j++) {
					if (targets[j][0] == classes[i]) count = count + 1;
			}
			//cout << count << endl;
			vector<float> sum_perceptron, avg_perceptron;
			for (int k = 0; k < seed_weights.back()[0].size(); k++) sum_perceptron.push_back(0);
			//filter indices
			vector<float> filter_indices;
			//cout << temp.size() << endl;
			for (int ind = 0; ind < temp.size(); ind++) {
				if (temp[ind] == classes[i]) filter_indices.push_back(ind);
			}
			//PrintVector(filter_indices);
			//Class_inputs
			vector<vector<float>> Class_inputs;
			for (int cinp = 0; cinp < filter_indices.size(); cinp++) {
				Class_inputs.push_back(inputs[filter_indices[cinp]]);
			}
			
			for (int inp = 0; inp < Class_inputs.size(); inp++) {
				vector<float> output_single = forward_propogate_seed(Class_inputs[inp]);
				for (int sw = 0; sw < seed_activations.back().size(); sw++) sum_perceptron[sw] += output_single[sw];
			}

			for (int s = 0; s < sum_perceptron.size(); s++) avg_perceptron.push_back((float)sum_perceptron[s] / count);

			vector<float> Error_list;
			for (int cimp = 0; cimp < Class_inputs.size(); cimp++) {
				int err = 0;
				vector<float> output_single = forward_propogate_seed(Class_inputs[cimp]);
				for (int o = 0; o < output_single.size(); o++) err += avg_perceptron[o] - output_single[o];
				Error_list.push_back(err);
			}
			//sort_indices = np.argsort(Error_list)
			vector<int> sorted_indices = sorted_index(Class_inputs.size(), Error_list);
			vector<vector<float>> Classes_sorted;
			//for (int is = 0; is < sorted_indices.size(); is++) cout << sorted_indices[is] << endl;
			for (int sc = 0; sc < sorted_indices.size(); sc++) Classes_sorted.push_back(Class_inputs[sorted_indices[sc]]);
			//PrintMatrix(Class_inputs);
			sorted_classes_list.push_back(Classes_sorted);
		}
		return sorted_classes_list;

	}

	vector<float> return_source_activations() {
		return activations.rbegin()[1];
	}

	float return_acc(vector<vector<float>> inputs, vector<vector<float>> targets) {

		vector<int> pred;
		for (int i = 0; i < inputs.size(); i++) {
			vector<float> output = forward_propogate(inputs[i]);
			//cout << output[0] << endl;
			if (output[0] >= 0.5) pred.push_back(1);
			else pred.push_back(0);
		}
		float acc = 0.0;
		int count = 0;
		for (int j = 0; j < pred.size(); j++) {
			//cout << pred[j] << endl;
			if (pred[j] == targets[j][0]) count += 1;
		}
		//cout << "acc: " << acc << endl;
		acc = (float)count / pred.size();
		cout << "acc: " << acc << endl;
		return acc;

	}

	void ANG_grow(vector<vector<float>> inputs, vector<vector<float>> targets, vector<int> hidden_layers) {
		//Init(hidden_layers, 1, inputs[0].size());
		//cout << weights.size() << endl;
		prime_base_network(inputs, targets, 10);
		//cout << weights.size() << endl;
		remove_temp_classifier();
		//cout << weights.size() << endl;
		add_destination_layer();
		//Print3D(weights);
		//cout << weights.size() << endl;
		add_class_layer(targets);
		//Print3D(weights);
		//cout << weights.size() << endl;
		float accuracy = 0.0;
		int percep = 0;
		vector<vector<float>> items = inputs, test_targets = targets;

		while (accuracy < 0.9) {
			vector<vector<vector<float>>> sorted_classes = extreme_member_classes(inputs, targets);
			//cout << weights.size() << endl;
			remove_temp_classifier();
			//cout << activations.back().size() << endl;
			for (int i = 0; i < sorted_classes.size(); i++) {
				vector<vector<float>> extremes;
				//PrintMatrix(sorted_classes[i]);
				extremes.push_back(sorted_classes[i][0]);
				extremes.push_back(sorted_classes[i].back());
				//PrintMatrix(extremes);
				for (int ext = 0; ext < extremes.size(); ext++) {
					vector<float> output = forward_propogate(extremes[ext]);
					vector<float> fp_output = return_source_activations();
					float sum_op = 0.0;
					for (int fpo = 0; fpo < fp_output.size(); fpo++) sum_op += fp_output[fpo];
					int len_op = fp_output.size();
					float average = (float)sum_op / len_op;
					float sum_avg = 0.0;
					for (int sa = 0; sa < fp_output.size(); sa++) sum_avg += (average - fp_output[sa]) * (average - fp_output[sa]);
					float sd = stdev(fp_output);
					int x = 1;
					//cout << "percep " << percep << endl;
					if(percep<activations.back().size()) {
						activations.back()[percep] = 0;
					}
					else {
						vector<vector<float>> temp_weights;
						vector<float> tw;
						for (int w = 0; w < weights.back().size(); w++) {
							tw = weights.back()[w];
							tw.push_back((float)rand() / RAND_MAX);
							temp_weights.push_back(tw);
						}
						//PrintMatrix(temp_weights);
						weights.back() = temp_weights;
						//PrintMatrix(weights.back());
						vector<vector<float>> temp_derivatives;
						vector<float> td;
						for (int d = 0; d < derivatives.back().size(); d++) {
							td = derivatives.back()[d];
							td.push_back(0);
							temp_derivatives.push_back(td);
						}
						derivatives.back() = temp_derivatives;
						activations.back().push_back(0);
					}
					percep = percep + 1;
					for (int conn = 0; conn < fp_output.size(); conn++) {
						if (fp_output[conn] >= x * sd || fp_output[conn] < -x * sd) cout << "This is a critical connection" << endl;
						else {
							cout << "This is not a critical connection" << endl;
							cout << fp_output.size() << endl;
							for (int wl = 0; wl < weights.back().size(); wl++)
								for(int c = 0; c< weights.back()[0].size(); c++)
									weights.back()[wl][c] = 0;
						}
					}
				}
			}
			cout << "I am here" << endl;
			add_class_layer(targets);
			cout << weights.back().size() << endl;
			train(inputs, targets, 50, 0.5);
			cout << "I am here" << endl;
			accuracy = return_acc(items, test_targets);
			vector<vector<float>> extreme_members;
			for (int scl = 0; scl < sorted_classes.size(); scl++) {
				extreme_members.push_back(sorted_classes[scl][0]);
				extreme_members.push_back(sorted_classes[scl].back());
			}
			vector<int> rm_index;
			for (int em = 0; em < extreme_members.size(); em++) {
				for (int pos = 0; pos < inputs.size(); pos++) {
					if (extreme_members[em] == inputs[pos]) {
						//rm_index.push_back(pos);
						inputs.erase(inputs.begin() + pos);
						targets.erase(targets.begin() + pos);
						break;
					}
				}
			}
			//Work on this
			/*for (auto rm : rm_index) {
				inputs.erase(inputs.begin() + rm);
				targets.erase(targets.begin() + rm);
			}*/
			cout << "accuracy: " << accuracy << endl;
		}
		//cout << "activations size: " << activations.size() << endl;
	}
	/*
	void save(ostringstream& oss)
	{
		boost::archive::binary_oarchive oa(oss);
		oa&* (this);
	}

	void load(istringstream& iss)
	{
		//std::string str_data = oss.str();
		//std::istringstream iss(str_data);
		boost::archive::binary_iarchive ia(iss);
		ia&* (this);
	}
	*/
	void save_weights_der(vector<vector<vector<float>>> Matrix, string name) {
		ofstream output_file(name + ".txt");
		ostream_iterator<float> output_iterator(output_file, ",");
		for (int i = 0; i < Matrix.size(); i++) {

			for (int j = 0; j <Matrix[i].size(); j++) {
				//cout << j;
				copy(Matrix[i][j].begin(), Matrix[i][j].end() - 1, output_iterator);
				output_file << Matrix[i][j].back();
				output_file << "\n";
			}
			output_file << "#\n";
		}

		output_file.close();

	}

	void save_activations(vector<vector<float>> Matrix, string name) {

		ofstream output_file(name + ".txt");
		ostream_iterator<float> output_iterator(output_file, ",");

		for (int j = 0; j < Matrix.size(); j++) {
			//cout << j;
			copy(Matrix[j].begin(), Matrix[j].end() - 1, output_iterator);
			output_file << Matrix[j].back();
			output_file << "\n";
		}

		output_file.close();
	
	}

	void read_weights_der(vector<vector<vector<float>>>& Matrix, string name) {

		std::ifstream in(name + ".txt");
		std::string str;
		vector<vector<vector<float>>> weights;
		vector<vector<float>> st;
		while (in.good())
		{
			string line;
			vector<float> row;
			getline(in, line, '\n');
			if (line == "") break;
			if (line == "#") {
				weights.push_back(st);
				st = {};
				//cout <<"size" << st.size() << endl;
				continue;
			}
			//cout << "size" << st.size() << endl;
			stringstream ss(line);

			while (ss.good()) {
				//std::string::size_type sz;
				string sbstr;
				getline(ss, sbstr, ',');
				float val = stof(sbstr);
				row.push_back(val);
			}

			st.push_back(row);

		}

		Matrix = weights;
	}

	void read_activations(vector<vector<float>>& Act, string name) {
		vector<vector<float>> A;
		ifstream ip(name + ".txt");
		while (ip.good()) {
			vector<float> st;
			string line;
			getline(ip, line, '\n');
			if (line == "") break;
			stringstream ss(line);
			//cout << line << endl;
			while (ss.good()) {
				std::string::size_type sz;
				string sbstr;
				getline(ss, sbstr, ',');
				float val = stof(sbstr);
				st.push_back(val);
			}
			A.push_back(st);
		}

		Act = A;

	}

	void save_model(string name) {
		save_weights_der(weights, name + "_weights");
		save_weights_der(derivatives, name + "_derivatives");
		save_weights_der(seed_weights, name + "_seed_weights");
		save_weights_der(seed_derivatives, name + "_seed_derivatives");
		save_activations(activations, name + "_activations");
		save_activations(seed_activations, name + "_seed_activations");
		
	}

	void load_model(string name) {
		read_weights_der(weights, name + "_weights");
		read_weights_der(seed_weights, name + "_seed_weights");
		read_weights_der(derivatives, name + "_derivatives");
		read_weights_der(seed_derivatives, name + "_seed_derivatives");
		read_activations(activations, name + "_activations");
		read_activations(seed_activations, name + "_seed_activations");
	}

};

//new stuff
void read_files(string filename, vector<vector<float>>& inputs, vector<vector<float>>& targets) {
	ifstream ip(filename);

	if (!ip.is_open()) cout << "Error" << endl;
	else cout << "Yay" << endl;

	ip.ignore(500, '\n');
	//vector<vector<float>> inputs;
	vector<string> st;
	while (ip.good()) {
		vector<float> st;
		string line;
		getline(ip, line, '\n');
		if (line == "") break;
		stringstream ss(line);
		//cout << line << endl;
		while (ss.good()) {
			std::string::size_type sz;
			string sbstr;
			getline(ss, sbstr, ',');
			float val = stof(sbstr);
			st.push_back(val);
		}
		//cout << st.size() << endl;
		inputs.push_back(st);
	}
	ip.close();
	cout <<"number of rows: " << inputs.size() << endl;
	//vector<vector<float>> targets;
	for (int tar = 0; tar < inputs.size(); tar++) {
		vector<float> t;
		t.push_back(inputs[tar].back());
		targets.push_back(t);
		inputs[tar].pop_back();
	}
	return;
}

ANG Train_ng(vector<vector<float>> inputs, vector<vector<float>> targets, vector<int> hidden_layers) {

	ANG ang(hidden_layers, targets[0].size(), inputs[0].size());
	ang.ANG_grow(inputs, targets, hidden_layers);

	return ang;
}

vector<int> test(ANG ang, vector<vector<float>> inputs) {
	vector<int> pred;

	for (int i = 0; i < inputs.size(); i++) {
		vector<float> output = ang.forward_propogate(inputs[i]);
		//cout << output[0] << endl;
		if (output[0] >= 0.5) pred.push_back(1);
		else pred.push_back(0);
	}
	return pred;
}

int main()
{
	cout << "Hello \n";
	ANG ang;
	//vector<vector<float>> inputs, targets;
	//string filename = "D:/Tests/Titanic/train_titanic.csv";
	//read_files(filename, inputs, targets);
	//ang.PrintMatrix(inputs);
	vector<vector<float>>  i = { {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , {0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , { 0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , { 0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , { 0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , { 0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0} };
	vector<vector<float>> t = { {1} , {0}, {0} , {0}, {1}, {1}, {0} , {0}, {1}, {0}, {1} , {0}, {1} , {0}, {1}, {1}, {0} , {0}, {1}, {0}, {1} , {0}, {1} , {0}, {1}, {1}, {0} , {0}, {1}, {0}, {1} , {0}, {1} , {0}, {1}, {1}, {0} , {0}, {1}, {0}, {1} , {0}, {1} , {0}, {1}, {1}, {0} , {0}, {1}, {0} };
	ang = Train_ng(i, t, {3,4});
	//std::ostringstream oss;
	//ang.save(oss);
	ang.save_model("saveloadtest");
	//string s = oss.str();
	ANG ang2;
	//std::istringstream iss;
	//iss.str(s);
	//ang2.load(iss);
	ang2.load_model("saveloadtest");
	/*
	cout << "weights" << endl;
	ang2.Print3D(ang2.weights);
	cout << "seed_weights" << endl;
	ang2.Print3D(ang2.seed_weights);
	cout << "derivatives" << endl;
	ang2.Print3D(ang2.derivatives);
	cout << "seed_derivatives" << endl;
	ang2.Print3D(ang2.seed_derivatives);
	cout << "activations" << endl;
	ang2.PrintMatrix(ang2.activations);
	cout << "seed_activations" << endl;
	ang2.PrintMatrix(ang2.seed_activations);
	*/
	vector<int> pred1 = test(ang, i);
	for (int i = 0; i < pred1.size(); i++) cout << pred1[i] << " ";
	cout << endl;
	vector<int> pred = test(ang2, i);
	for (int i = 0; i < pred.size(); i++) cout << pred[i] << " ";
	//ang.ANG_grow(i, t, {3,3});
	//ang.create_seed();
	//ang.create_temp_classifier_seed(t);
	//cout << ang.seed_activations.size() << endl;
	//ang.train_seed(i, t, 10, 0.1);
	//vector<vector<vector<float>>> emc = ang.extreme_member_classes(i, t);
	//cout << emc.size() << endl;
	//ang.add_class_layer(t);
	//ang.train(i, t, 10, 1.0);
	//ang.PrintVector(ang.activations.back());
	//vector<vector<float>> p = ang.dot_new(a, b);
	//ang.PrintMatrix(p);
	//vector<vector<float>> trans = ang.transpose(b);
	//ang.PrintMatrix(trans);
	//vector<float> t = {1 ,0, 1, 4, 5};
	//sort(t.begin(), t.end());
	//vector<float>::iterator ip = std::unique(t.begin(), t.end());
	//t.resize(std::distance(t.begin(), ip));
	//ang.PrintVector(t);
	//vector<int> ind;
	//ind = ang.sorted_index(5, t);
	//for (int i = 0; i < ind.size(); i++) cout << ind[i] << " ";
	return 0;
}
