
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "ANG.h"
#include <iostream>
using namespace std;


ANG Train(vector<vector<float>> inputs, vector<vector<float>> targets, vector<int> hidden_layers) {
	ANG ang(hidden_layers, targets[0].size(), inputs[0].size());
	ang.ANG_grow(inputs, targets);
	return ang;
}

vector<int> test_ang(vector<int> hidden_layers, vector<vector<float>> inputs, vector<vector<float>> targets) {
	ANG ang;
	ang = Train(inputs, targets, hidden_layers);
	vector<int> pred;

	for (int i = 0; i < inputs.size(); i++) {
		vector<float> output = ang.forward_propogate(inputs[i]);
		//cout << output[0] << endl;
		if (output[0] >= 0.5) pred.push_back(1);
		else pred.push_back(0);
	}
	//ang.Print3D(ang.weights);
	//Test saving
	ang.save_model("pybindsave");
	ANG ang2;
	vector<int> pred2;
	ang2.load_model("pybindsave");
	for (int i = 0; i < inputs.size(); i++) {
                vector<float> output = ang2.forward_propogate(inputs[i]);
                //cout << output[0] << endl;
                if (output[0] >= 0.5) pred2.push_back(1);
                else pred2.push_back(0);
        }
	
	for (int j = 0; j< pred2.size(); j++){
		cout << pred2[j] << "\t";
	}
	return pred;

	//for (int j = 0; j < pred.size(); j++) {
	//	cout << pred[j] << '\t';
	//}
}

//int main() {
	//ANG ang({3,3}, 1, 10);
	//ang.PrintVector({ 1,2,3 });
	//vector<vector<float>>  i = { {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , {0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , { 0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , { 0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , { 0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1} , { 0,0,0} ,{1,1,1}, {1,1,1} , {0,0,0}, {0,0,0} ,{1,1,1} , {0,0,0} };
	//vector<vector<float>> t = { {1} , {0}, {0} , {0}, {1}, {1}, {0} , {0}, {1}, {0}, {1} , {0}, {1} , {0}, {1}, {1}, {0} , {0}, {1}, {0}, {1} , {0}, {1} , {0}, {1}, {1}, {0} , {0}, {1}, {0}, {1} , {0}, {1} , {0}, {1}, {1}, {0} , {0}, {1}, {0}, {1} , {0}, {1} , {0}, {1}, {1}, {0} , {0}, {1}, {0} };
	//test_ang({ 3,3 }, i, t);
//}

class Ang {
	ANG ang;
public:
	Ang(vector<int> hidden_layers, int t_size, int i_size) {
		ANG ang2(hidden_layers, t_size, i_size);
		ang = ang2;
	}

	ANG train(vector<vector<float>> inputs, vector<vector<float>> targets) {
		ang.ANG_grow(inputs, targets);
		return ang;
	}

	void save(string s) {
		ang.save_model(s);
	}

	ANG load(string s) {
		ANG loaded_ang;
		loaded_ang.load_model(s);
		return loaded_ang;
	}

	vector<int> test(vector<vector<float>> inp) {
		vector<int> pred;
		for (int i = 0; i < inp.size(); i++) {
			vector<float> output = ang.forward_propogate(inp[i]);
			//cout << output[0] << endl;
			if (output[0] >= 0.5) pred.push_back(1);
			else pred.push_back(0);
		}
		return pred;
	}

};


namespace py = pybind11;
/*
class TestClass {
	float multiplier1;
	float multiplier2;
public:
	TestClass(float m, float n) { multiplier1 = m;
				      multiplier2 = n;}
	float multiply(float input) {
		return multiplier1 * input * multiplier2;
	}
	std::vector<float> multiply_lists(std::vector<float> a, std::vector<float> b){
		for(int i = 0; i<a.size(); i++) a[i] = a[i] * b [i];
		return a;
}
};

*/


/*
float add(float a, float b) {
	return a + b;
}
*/

PYBIND11_MODULE(example1, m) {

	m.doc() = "first test of pybind11 on ANG";

	m.def("test_ang", &test_ang, "Function to test Artificial NeuroGenesis");

	py::class_<Ang>(m, "ClassTest")
		.def(py::init<vector<int>, int, int>())
		.def("Train", &Ang::train)
		.def("save", &Ang::save)
		.def("load", &Ang::load)
		.def("test", &Ang::test)
		;
	}

	/*
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    py::class_<TestClass> (m,"PyTest")
				.def(py::init<float, float>())
				.def("multiply", &TestClass::multiply)
				.def("multiply_list", &TestClass::multiply_lists)
				;
	}
	*/ 






