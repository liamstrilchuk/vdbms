#pragma once
#include <vector>
#include <cstdint>
#include <string>

using Embedding = std::vector<float>;

struct EdgeList {
	uint32_t source_id;
	std::vector<uint32_t> neighbours;
};

struct VectorNode {
	uint32_t id;
	Embedding vector_data;
};