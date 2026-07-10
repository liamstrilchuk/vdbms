#include "../include/graph_index.h"

GraphIndex::GraphIndex() {

}

void GraphIndex::add_edge(uint32_t source, uint32_t dest) {
	if (this->adjacency_list.find(source) == this->adjacency_list.end()) {
		this->adjacency_list.insert({ source, {} });
	}

	this->adjacency_list.find(source)->second.push_back(dest);
}