#include "../include/graph_index.h"
#include <algorithm>
#include <iostream>

GraphIndex::GraphIndex() {

}

void add_if_not_exists(std::unordered_map<uint32_t, std::vector<uint32_t>>* map, uint32_t key, uint32_t value) {
	if (map->find(key) == map->end()) {
		map->insert({ key, {} });
	}

	auto vec = &map->find(key)->second;

	if (std::find(vec->begin(), vec->end(), value) == vec->end()) {
		vec->push_back(value);
	}
}

void GraphIndex::add_edge(uint32_t source, uint32_t dest) {
	add_if_not_exists(&this->outbound_edges, source, dest);
	add_if_not_exists(&this->inbound_edges, dest, source);
}