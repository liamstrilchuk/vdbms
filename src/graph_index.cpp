#include "../include/graph_index.h"

GraphIndex::GraphIndex() {

}

void GraphIndex::add_edges(uint32_t source_id, const std::vector<uint32_t>& targets) {
	this->adjacency_list.push_back({ source_id, targets });
}