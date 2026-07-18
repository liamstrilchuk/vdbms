#include "../include/hnsw_graph.h"

const uint32_t BFS_STEPS = 4;

void HNSWGraph::add_to_queried_indexes(uint32_t id) {
	this->last_queried_indexes.push(id);
	this->nodes[id].access_count++;

	// double ratio = (double) this->nodes[id].access_count * (double) this->nodes.size() / (double) this->last_queried_indexes.size();
	if (this->nodes[id].access_count > 20 && !this->nodes[id].is_suppressed) {
		// uint32_t promote_to = std::floor(std::log(ratio) / std::log(constants::HNSW_M));
		// this->promote(this->nodes[id].base_index, promote_to);
		auto nbh = this->run_neighborhood_bfs(this->nodes[id].base_index, BFS_STEPS);
		// std::cout << "Cluster identified: ";
		// print_vector(nbh);
		auto closest = this->find_cluster_centroid(nbh, this->nodes[id].base_index);
		// std::cout << "Centroid is " << closest << std::endl;
		this->promote_centroid(closest, nbh);
	}

	if (this->last_queried_indexes.size() > constants::HNSW_QUERIED_IDS_MAX_SIZE) {
		this->nodes[this->last_queried_indexes.front()].access_count--;
		this->last_queried_indexes.pop();
	}
}

void HNSWGraph::promote_centroid(uint32_t centroid, std::vector<uint32_t> cluster) {
	uint32_t cluster_sum = 0;

	for (uint32_t node : cluster) {
		cluster_sum += this->nodes[this->layers[0][node].node_index].access_count;
	}

	double ratio = (double) cluster_sum * (double) this->nodes.size()
		/ (double) constants::HNSW_QUERIED_IDS_MAX_SIZE / (double) std::min(cluster.size(), (size_t) 3);

	if (ratio < 5) {
		std::cout << "Decided not to promote: ratio of " << ratio << std::endl;
		return;
	}

	// std::cout << "Ratio of: " << ratio << std::endl;

	uint32_t promote_to = std::floor(std::log(ratio) / std::log(constants::HNSW_M)) * 2;
	this->promote(centroid, promote_to);

	for (uint32_t node : cluster) {
		this->nodes[this->layers[0][node].node_index].is_suppressed = true;
	}

	this->nodes[this->layers[0][centroid].node_index].is_suppressed = false;
}

uint32_t HNSWGraph::find_cluster_centroid(std::vector<uint32_t> cluster, uint32_t start_node) {
	std::vector<float> combined_vec(384, 0);

	for (uint32_t elem : cluster) {
		const Embedding& vec_data = this->nodes[this->layers[0][elem].node_index].vector_data;
		for (int i = 0; i < vec_data.size(); i++) {
			combined_vec[i] += vec_data[i] / static_cast<float>(cluster.size());
		}
	}

	return this->find_closest_in_layer(combined_vec, start_node, this->layers[0], false);
}

std::vector<uint32_t> HNSWGraph::run_neighborhood_bfs(uint32_t base_index, uint32_t steps) {
	if (steps == BFS_STEPS) {
		this->prepare_visited_tracker(this->layers[0].size());
	}

	this->visited_tracker[base_index] = this->current_visited_version;
	std::vector<uint32_t> cluster;
	cluster.push_back(base_index);

	if (steps == 0) {
		return cluster;
	}

	for (int idx = 0; idx < constants::HNSW_M; idx++) {
		int32_t nid = this->layers[0][base_index].hnsw_neighbors[idx];
		if (nid == -1 || this->visited_tracker[nid] == this->current_visited_version) {
			continue;
		}

		const VectorNode& node = this->nodes[this->layers[0][nid].node_index];
		if (node.access_count >= 5) {
			auto res = this->run_neighborhood_bfs(nid, steps - 1);
			cluster.insert(cluster.end(), res.begin(), res.end());
		}
	}

	return cluster;
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

		uint32_t closest = this->find_closest_in_layer(node.vector_data, 0, this->layers[current_level], false);
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
