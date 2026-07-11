#pragma once
#include <vector>
#include <unordered_map>
#include <random>
#include "types.h"

class HNSWGraph final {
public:
	HNSWGraph();
	void add_node(uint32_t node_id, Embedding& vec);

private:
	std::unordered_map<uint32_t, uint32_t> id_to_index;
	std::vector<VectorNode> nodes;
	std::vector<std::vector<HNSWNode>> layers;
	bool is_initialized = false;

	std::mt19937 gen;
	std::uniform_real_distribution<double> dis;

	void create_layers();
	uint32_t get_node_layer();
	HNSWNode* find_closest_in_layer(Embedding& vec, HNSWNode& start_node, std::vector<HNSWNode>& layer) const;
	double calculate_cosine_similarity(const Embedding& vec1, const Embedding& vec2) const;
};