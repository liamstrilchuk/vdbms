#include "../include/hnsw_graph.h"

void HNSWGraph::add_to_queried_indexes(uint32_t id) {
	this->last_queried_indexes.push(id);
	this->nodes[id].access_count++;

	double ratio = (double) this->nodes[id].access_count * (double) this->nodes.size() / (double) this->last_queried_indexes.size();
	if (this->nodes[id].access_count > 5 && ratio > 100) {
		uint32_t promote_to = std::floor(std::log(ratio) / std::log(constants::HNSW_M));
		this->promote(this->nodes[id].base_index, promote_to);
	}

	if (this->last_queried_indexes.size() > constants::HNSW_QUERIED_IDS_MAX_SIZE) {
		this->nodes[this->last_queried_indexes.front()].access_count--;
		this->last_queried_indexes.pop();
	}
}

void HNSWGraph::promote(uint32_t base_index, uint32_t level) {
	VectorNode& node = this->nodes[this->layers[0][base_index].node_index];
	uint32_t current_level = 0;
	uint32_t current_index = base_index;

	while (this->layers[current_level][current_index].higher_level_index != -1) {
		uint32_t next_index = this->layers[current_level][current_index].higher_level_index;
		current_level++;
		current_index = next_index;
	}

	if (current_level < level && current_level < this->layers.size() - 1) {
		std::cout << "Promoting " << node.id << " from " << current_level << " to " << level << std::endl;
	}

	while (current_level < level && current_level < this->layers.size() - 1) {
		current_level++;

		uint32_t closest = this->find_closest_in_layer(node.vector_data, 0, this->layers[current_level]);
		auto ef_const = this->run_ef_construction(node.vector_data, closest, this->layers[current_level], constants::HNSW_EF_CONSTRUCTION);
		auto pruned = this->prune_ef_construction(node.vector_data, ef_const, this->layers[current_level]);

		this->layers[current_level].emplace_back(HNSWNode{
			.node_index = this->layers[current_level - 1][current_index].node_index,
			.hnsw_neighbors = {-1},
			.lower_level_index = current_index,
			.higher_level_index = -1
		});
		this->layers[current_level - 1][current_index].higher_level_index = this->layers[current_level].size() - 1;
		current_index = this->layers[current_level].size() - 1;

		std::copy(pruned.begin(), pruned.end(), this->layers[current_level].back().hnsw_neighbors);

		for (int idx = 0; idx < constants::HNSW_M; idx++) {
			if (pruned[idx] != -1) {
				this->create_reverse_connection(pruned[idx], current_index, this->layers[current_level]);
			}
		}
	}
}
