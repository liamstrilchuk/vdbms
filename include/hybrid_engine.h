#pragma once
#include <memory>
#include <unordered_set>
#include "types.h"
#include "vector_index.h"
#include "graph_index.h"

class HybridEngine final {
public:
	HybridEngine();

	void insert_record(uint32_t id, const Embedding& vec);
	void insert_edge(std::pair<uint32_t, uint32_t> edge);

	uint32_t find_nearest_neighbour(const Embedding& vec) const;

private:
	std::unordered_set<uint32_t> used_ids;

	std::unique_ptr<VectorIndex> vector_index;
	std::unique_ptr<GraphIndex> graph_index;
};