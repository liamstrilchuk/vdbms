#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "constants.h"

using Embedding = std::vector<float>;

struct VectorNode {
	uint32_t id;
	uint32_t access_count;
	bool is_suppressed;
	uint32_t base_index;
	Embedding vector_data;
	std::vector<uint32_t> neighbors;
};

struct HNSWNode {
	uint32_t node_index;
	int32_t hnsw_neighbors[constants::HNSW_M];
	uint32_t lower_level_index;
	int32_t higher_level_index;
};