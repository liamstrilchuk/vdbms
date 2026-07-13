#pragma once
#include <vector>
#include <unordered_map>
#include <random>
#include "types.h"

using ef_pair_list = std::vector<std::pair<uint32_t, double>>;

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
	ef_pair_list run_ef_construction(
		Embedding& vec, uint32_t closest, std::vector<HNSWNode>& layer
	) const;
	int32_t* prune_ef_construction(Embedding& vec, ef_pair_list& ef_construction, std::vector<HNSWNode>& layer) const;
	uint32_t find_closest_in_layer(Embedding& vec, uint32_t start_node, std::vector<HNSWNode>& layer) const;
	double calculate_cosine_similarity(const Embedding& vec1, const Embedding& vec2) const;
};