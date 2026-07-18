#pragma once
#include <iostream>
#include <vector>

template <typename T>
void print_vector(const std::vector<T>& vec) {
	for (size_t i = 0; i < vec.size(); i++) {
		std::cout << vec[i];
		if (i < vec.size() - 1) {
			std::cout << ", ";
		}
	}
	std::cout << std::endl;
}