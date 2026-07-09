#include "../include/hybrid_engine.h"

HybridEngine::HybridEngine() {
	this->graph_index = std::make_unique<GraphIndex>();
	this->vector_index = std::make_unique<VectorIndex>();
}

void HybridEngine::insert_record(uint32_t id, const Embedding& vec, std::vector<uint32_t> linked_ids) {
	this->graph_index->add_edges(id, linked_ids);
	this->vector_index->add_vector(id, vec);
	this->node_count++;
}

uint32_t HybridEngine::find_nearest_neighbour(const Embedding& vec) const {
	return this->vector_index->find_nearest_neighbour(vec);
}