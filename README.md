# Artificial-Neurogenesis
Implementation of the paper, Artificial Neurogenesis, by Mixter et al.

Artificial Neurogenesis allows for the creation of scalable models that grow rather than prune pre-trained models. Request access if required.


# Steps to create and use python bindings. 

clone the pybind repository into this folder

mkdir build

cd build

cmake .. && make

cd out of build

(Alter the contents of the CMAKE file. It is currently set to compile both the pybind11test.cpp file and the ANG.cpp file)

Open a python session in this folder and run:

from build.example1 (or any other name) import *
