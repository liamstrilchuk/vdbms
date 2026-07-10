#include "../include/hybrid_engine.h"

HybridEngine::HybridEngine() {
	this->graph_index = std::make_unique<GraphIndex>();
	this->vector_index = std::make_unique<VectorIndex>();
}

void HybridEngine::insert_record(uint32_t id, const Embedding& vec) {
	if (this->used_ids.count(id) > 0) {
		return;
	}
	
	this->vector_index->add_vector(id, vec);
	this->used_ids.insert(id);
}

void HybridEngine::insert_edge(std::pair<uint32_t, uint32_t> edge) {
	this->graph_index->add_edge(edge.first, edge.second);
}

uint32_t HybridEngine::find_nearest_neighbour(const Embedding& vec) const {
	return this->vector_index->find_nearest_neighbour(vec);
}