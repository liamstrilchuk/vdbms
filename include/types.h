#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "constants.h"

using Embedding = std::vector<float>;

struct EdgeList {
	uint32_t source_id;
	std::vector<uint32_t> neighbours;
};

struct VectorNode {
	uint32_t id;
	Embedding vector_data;
	std::vector<uint32_t> neighbors;
};

struct HNSWNode {
	uint32_t node_index;
	int32_t hnsw_neighbors[constants::HNSW_M];
	uint32_t lower_level_index;
};