#pragma once
#include <memory>
#include "types.h"
#include "vector_index.h"
#include "graph_index.h"

class HybridEngine final {
public:
	HybridEngine();

	void insert_record(uint32_t id, const Embedding& vec, const std::vector<uint32_t> linked_ids);

	uint32_t find_nearest_neighbour(const Embedding& vec) const;

private:
	uint32_t node_count = 0;
	
	std::unique_ptr<VectorIndex> vector_index;
	std::unique_ptr<GraphIndex> graph_index;
};