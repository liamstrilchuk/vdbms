#pragma once
#include <vector>
#include <unordered_map>
#include "types.h"

class HNSWGraph final {
public:
	HNSWGraph();
	void add_node(uint32_t node_id, Embedding& vec);

private:
	std::unordered_map<uint32_t, uint32_t> id_to_index;
	std::vector<HNSWNode> nodes;
	std::vector<std::vector<int>> layers;

	void create_layers();
};