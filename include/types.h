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
};

struct HNSWNode {
	Embedding vector_data;
	std::vector<uint32_t> neighbors;
	uint32_t hnsw_neighbors[constants::HNSW_M];
};