#include "../include/hnsw_graph.h"
#include <cmath>
#include <iostream>

HNSWGraph::HNSWGraph() : gen(std::random_device{}()), dis(0.0, 1.0) {
	this->create_layers();
}

void HNSWGraph::add_node(uint32_t node_id, Embedding& vec) {
	if (this->id_to_index.find(node_id) != this->id_to_index.end()) {
		return;
	}

	this->nodes.push_back({ node_id, vec, {} });
	this->id_to_index.insert({ node_id, this->nodes.size() - 1 });

	if (!this->is_initialized) {
		this->is_initialized = true;
		uint32_t last_node_index = 0;

		for (auto& layer : this->layers) {
			layer.emplace_back(HNSWNode{
				.node_index = static_cast<uint32_t>(this->nodes.size() - 1),
				.hnsw_neighbors = {},
				.lower_level_index = last_node_index
			});
			last_node_index = layer.size() - 1;
		}

		return;
	}

	int current_layer = this->layers.size() - 1;
	uint32_t max_add_layer = this->get_node_layer();

	HNSWNode* current = &this->layers[current_layer][0];

	while (current_layer > -1) {
		HNSWNode* closest = this->find_closest_in_layer(vec, *current);
		current_layer--;
	}
}

HNSWNode* HNSWGraph::find_closest_in_layer(Embedding& vec, HNSWNode& start_node) const {

}

uint32_t HNSWGraph::get_node_layer() {
	if (!this->is_initialized) {
		return this->layers.size() - 1;
	}

	double rand = this->dis(this->gen);
	uint32_t layer = 0;

	while (rand < std::exp(-(layer / (1 / std::log(constants::HNSW_M)))) && layer < this->layers.size() - 1) {
		layer++;
	}

	return layer - 1;
}

void HNSWGraph::create_layers() {
	int layer = 0;

	while (true) {
		double layer_prob = std::exp(-(layer / (1 / std::log(constants::HNSW_M))));

		if (layer_prob < constants::HNSW_MIN_LAYER_PROB) {
			break;
		}

		this->layers.push_back({});
		layer++;
	}
}