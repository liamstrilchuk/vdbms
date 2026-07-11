#include "../include/hnsw_graph.h"
#include <cmath>
#include <iostream>

HNSWGraph::HNSWGraph() {
	this->create_layers();
}

void HNSWGraph::add_node(uint32_t node_id, Embedding& vec) {

}

void HNSWGraph::create_layers() {
	int layer = 0;

	while (true) {
		double layer_prob = std::exp(-layer / (1 / std::log(constants::HNSW_M)));

		if (layer_prob < constants::HNSW_MIN_LAYER_PROB) {
			break;
		}

		this->layers.push_back({});
		layer++;
	}
}