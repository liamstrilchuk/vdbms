#include <chrono>
#include <utility>
#include <memory>
#include <iostream>
#include <iomanip>
#include "../include/hnsw_graph.h"

std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<float> dis(-1.0, 1.0);

std::vector<float> make_random_embedding(int size);

int main() {
	auto graph = std::make_unique<HNSWGraph>();
	std::vector<std::vector<float>> embeddings;

	const int NUM_NODES = 1000;

	auto add_start = std::chrono::steady_clock::now();

	for (int i = 0; i < NUM_NODES; i++) {
		std::vector<float> embedding = make_random_embedding(384);
		embeddings.push_back(embedding);
		graph->add_node(i, embedding);
	}

	auto add_end = std::chrono::steady_clock::now();
	auto add_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(add_end - add_start).count();

	int recalled1 = 0;

	auto query_start = std::chrono::steady_clock::now();

	for (int i = 0; i < NUM_NODES; i++) {
		auto results = graph->search(embeddings[i], 5);

		if (results.size() > 0 && results[0] == i) {
			recalled1++;
		}
	}

	auto query_end = std::chrono::steady_clock::now();
	auto query_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count();

	float recall1 = recalled1 / static_cast<float>(NUM_NODES) * 100;

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Recall Values" << std::endl;
	std::cout << "Recall@1: " << recall1 << std::endl;

	float time_per_add = static_cast<float>(add_elapsed) / static_cast<float>(NUM_NODES);
	float time_per_query = static_cast<float>(query_elapsed) / static_cast<float>(NUM_NODES);
	std::cout << std::endl << "Performance" << std::endl;
	std::cout << "Time per insertion: " << time_per_add << " ms" << std::endl;
	std::cout << "Time per query: " << time_per_query << " ms" << std::endl;
}

std::vector<float> make_random_embedding(int size) {
	std::vector<float> result;

	for (int i = 0; i < size; i++) {
		result.push_back(dis(gen));
	}

	return result;
}