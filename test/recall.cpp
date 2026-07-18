#include <chrono>
#include <utility>
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <regex>
#include <algorithm>
#include "../include/hnsw_graph.h"

std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<float> dis(-1.0, 1.0);

std::vector<float> make_random_embedding(int size);
std::vector<std::vector<float>> load_embeddings_from_dataset(std::string dataset_name);
int get_random_index(int n, double p);

const int NUM_NODES = 10000;

int main() {
	auto graph = std::make_unique<HNSWGraph>();
	std::vector<std::vector<float>> embeddings;

	auto add_start = std::chrono::steady_clock::now();

	embeddings = load_embeddings_from_dataset("examples/data/embeddings.csv");
	size_t num_iters = std::min(static_cast<int>(embeddings.size()), NUM_NODES);

	for (int i = 0; i < num_iters; i++) {
		graph->add_node(i, embeddings[i]);
	}

	auto add_end = std::chrono::steady_clock::now();
	auto add_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(add_end - add_start).count();

	int recalled1 = 0;

	graph->step_count = 0;
	auto query_start = std::chrono::steady_clock::now();

	int NUM_QUERIES = 100000;

	auto results1 = graph->search(embeddings[embeddings.size() / 2], 5);

	for (int i = 0; i < NUM_QUERIES; i++) {
		// auto results = graph->search(embeddings[i], 5);
		// int index = (get_random_index(embeddings.size(), 0.005) + embeddings.size() / 2) % embeddings.size();
		int index = results1[i % results1.size()];
		auto results = graph->search(embeddings[index], 1);

		if (results.size() > 0 && results[0] == index) {
			recalled1++;
		}
	}

	auto query_end = std::chrono::steady_clock::now();
	auto query_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count();

	float recall1 = recalled1 / static_cast<float>(NUM_QUERIES) * 100;

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "RECALL VALUES" << std::endl;
	std::cout << "Recall@1: " << recall1 << std::endl;

	float time_per_add = static_cast<float>(add_elapsed) / static_cast<float>(num_iters);
	float time_per_query = static_cast<float>(query_elapsed) / static_cast<float>(NUM_QUERIES);
	std::cout << std::endl << "PERFORMANCE" << std::endl;
	std::cout << "Time per insertion: " << time_per_add << " ms" << std::endl;
	std::cout << "Time per query: " << time_per_query << " ms" << std::endl;

	double steps_per_op = static_cast<double>(graph->step_count) / static_cast<double>(NUM_QUERIES);
	std::cout << "Steps per query: " << steps_per_op << std::endl;
}

std::vector<float> make_random_embedding(int size) {
	std::vector<float> result;

	for (int i = 0; i < size; i++) {
		result.push_back(dis(gen));
	}

	return result;
}

std::vector<std::vector<float>> load_embeddings_from_dataset(std::string dataset_name) {
	std::ifstream file(dataset_name);

	if (!file.is_open()) {
		std::cerr << "Could not open " << dataset_name << std::endl;
		return {};
	}

	std::vector<std::vector<float>> data;
	std::string line;
	std::regex num_regex("-?\\d(?:\\.\\d+)?");

	while (std::getline(file, line) && data.size() < NUM_NODES) {
		std::sregex_iterator nums_begin(line.begin() + line.find("\""), line.end(), num_regex);
		std::sregex_iterator nums_end;
		data.push_back({});

		for (std::sregex_iterator i = nums_begin; i != nums_end; i++) {
			data[data.size() - 1].push_back(std::stof(i->str()));
		}
	}

	return data;
}

int get_random_index(int n, double p = 0.05) {
	std::geometric_distribution<int> d(p);

	int result;
	do {
		result = d(gen);
	} while (result > n - 1);

	return result;
}