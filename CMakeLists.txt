cmake_minimum_required(VERSION 3.1)
project(ArtificialNeurogenesis)
add_subdirectory(pybind11)
pybind11_add_module(example1 pybind11test.cpp ANG.cpp)
