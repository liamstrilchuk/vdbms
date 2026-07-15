#include "../include/hnsw_graph.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <queue>

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
				.hnsw_neighbors = {-1},
				.lower_level_index = last_node_index
			});
			last_node_index = layer.size() - 1;
		}

		return;
	}

	int current_layer = this->layers.size() - 1;
	uint32_t max_add_layer = this->get_node_layer();
	int32_t last_node_index = -1;
	uint32_t entry_point = 0;

	while (current_layer > -1) {
		uint32_t closest = this->find_closest_in_layer(vec, entry_point, this->layers[current_layer]);

		if (max_add_layer >= current_layer) {
			auto ef_const = this->run_ef_construction(vec, closest, this->layers[current_layer], constants::HNSW_EF_CONSTRUCTION);
			auto pruned = this->prune_ef_construction(vec, ef_const, this->layers[current_layer]);

			this->layers[current_layer].emplace_back(HNSWNode{
				.node_index = static_cast<uint32_t>(this->nodes.size() - 1),
				.lower_level_index = 0
			});

			if (last_node_index != -1) {
				this->layers[current_layer + 1][last_node_index].lower_level_index = this->layers[current_layer].size() - 1;
			}

			std::copy(pruned.begin(), pruned.end(), this->layers[current_layer].back().hnsw_neighbors);
			last_node_index = this->layers[current_layer].size() - 1;

			for (int idx = 0; idx < constants::HNSW_M; idx++) {
				if (pruned[idx] != -1) {
					this->create_reverse_connection(pruned[idx], last_node_index, this->layers[current_layer]);
				}
			}
		}

		if (current_layer > 0) {
			entry_point = this->layers[current_layer][closest].lower_level_index;
		}

		current_layer--;
	}
}

std::vector<uint32_t> HNSWGraph::search(Embedding& vec, int knn) {
	int current_layer = this->layers.size() - 1;
	uint32_t entry_point = 0;

	while (current_layer > 0) {
		uint32_t closest = this->find_closest_in_layer(vec, entry_point, this->layers[current_layer]);

		entry_point = this->layers[current_layer][closest].lower_level_index;

		current_layer--;
	}

	auto ef_const = this->run_ef_construction(
		vec, entry_point, this->layers[0],
		std::max(constants::HNSW_EF_SEARCH, knn)
	);

	std::vector<uint32_t> results;
	results.reserve(knn);
	
	for (int idx = ef_const.size() - 1; idx >= 0 && results.size() < knn; idx--) {
		results.push_back(this->nodes[ef_const[idx].first].id);
	}

	return results;
}

void HNSWGraph::create_reverse_connection(uint32_t node_id, uint32_t new_id, std::vector<HNSWNode>& layer) {
	Embedding& node_vector_data = this->nodes[layer[node_id].node_index].vector_data;
	ef_pair_list neighbor_list;
	neighbor_list.push_back({
		new_id,
		this->calculate_cosine_similarity(
			node_vector_data,
			this->nodes[layer[new_id].node_index].vector_data
		)
	});
	
	for (int idx = 0; idx < constants::HNSW_M; idx++) {
		if (layer[node_id].hnsw_neighbors[idx] != -1) {
			neighbor_list.push_back({
				layer[node_id].hnsw_neighbors[idx],
				this->calculate_cosine_similarity(
					node_vector_data,
					this->nodes[layer[layer[node_id].hnsw_neighbors[idx]].node_index].vector_data
				)
			});
		}
	}

	std::vector<int32_t> pruned = this->prune_ef_construction(node_vector_data, neighbor_list, layer);
	std::copy(pruned.begin(), pruned.end(), layer[node_id].hnsw_neighbors);
}

ef_pair_list HNSWGraph::run_ef_construction(
	Embedding& vec, uint32_t closest, std::vector<HNSWNode>& layer, const int node_count
) const {
	std::unordered_set<uint32_t> explored_nodes;

	struct QueueElement {
		uint32_t index;
		double score;

		bool operator<(const QueueElement& other) const {
			return this->score < other.score;
		}

		bool operator>(const QueueElement& other) const {
			return this->score > other.score;
		}
	};

	std::priority_queue<QueueElement> working_queue;
	std::priority_queue<
		QueueElement, std::vector<QueueElement>, std::greater<QueueElement>
	> candidate_pool;

	double closest_similarity = this->calculate_cosine_similarity(
		vec, this->nodes[layer[closest].node_index].vector_data
	);
	candidate_pool.push({ closest, closest_similarity });
	working_queue.push({ closest, closest_similarity });
	explored_nodes.insert(closest);

	while (working_queue.size() > 0) {
		QueueElement current = working_queue.top();
		QueueElement furthest = candidate_pool.top();
		working_queue.pop();

		if (current.score < furthest.score) {
			break;
		}

		int32_t* neighbors = layer[current.index].hnsw_neighbors;

		for (int idx = 0; idx < constants::HNSW_M; idx++) {
			if (neighbors[idx] != -1 && explored_nodes.find(neighbors[idx]) == explored_nodes.end()) {
				explored_nodes.insert(neighbors[idx]);

				const Embedding& nvec = this->nodes[layer[neighbors[idx]].node_index].vector_data;
				double similarity = this->calculate_cosine_similarity(vec, nvec);

				QueueElement worst = candidate_pool.top();
				bool should_add = candidate_pool.size() < node_count;

				if (candidate_pool.size() >= node_count && similarity > worst.score) {
					candidate_pool.pop();
					should_add = true;
				}

				if (should_add) {
					candidate_pool.push({ static_cast<uint32_t>(neighbors[idx]), similarity });
					working_queue.push({ static_cast<uint32_t>(neighbors[idx]), similarity });
				}
			}
		}
	}

	ef_pair_list result;

	while (!candidate_pool.empty()) {
		result.push_back({ candidate_pool.top().index, candidate_pool.top().score });
		candidate_pool.pop();
	}

	return result;
}

std::vector<int32_t> HNSWGraph::prune_ef_construction(
	Embedding& vec, ef_pair_list& ef_construction, std::vector<HNSWNode>& layer
) const {
	std::vector<int32_t> selected_M(constants::HNSW_M, -1);
	std::sort(ef_construction.begin(), ef_construction.end(), [](const auto& a, const auto& b) {
		return a.second < b.second;
	});
	int top_index = -1;

	while (top_index < constants::HNSW_M - 1 && ef_construction.size() > 0) {
		auto top = ef_construction.back();
		ef_construction.pop_back();
		bool should_add = true;

		for (int idx = 0; idx <= top_index; idx++) {
			double sim = this->calculate_cosine_similarity(
				this->nodes[layer[selected_M[idx]].node_index].vector_data,
				this->nodes[layer[top.first].node_index].vector_data
			);

			if (sim > top.second) {
				should_add = false;
				break;
			}
		}

		if (should_add) {
			selected_M[++top_index] = top.first;
		}
	}

	return selected_M;
}

uint32_t HNSWGraph::find_closest_in_layer(
	Embedding& vec, uint32_t start_node, std::vector<HNSWNode>& layer
) const {
	double best_similarity = this->calculate_cosine_similarity(
		vec, this->nodes[layer[start_node].node_index].vector_data
	);
	uint32_t best_index = start_node;

	bool has_changed = true;
	while (has_changed) {
		has_changed = false;
		int32_t* neighbors = layer[best_index].hnsw_neighbors;

		for (int idx = 0; idx < constants::HNSW_M; idx++) {
			uint32_t neighbor_idx = neighbors[idx];

			if (neighbors[idx] == -1) {
				continue;
			}

			double similarity = this->calculate_cosine_similarity(
				vec, this->nodes[layer[neighbor_idx].node_index].vector_data
			);

			if (similarity > best_similarity) {
				best_similarity = similarity;
				best_index = neighbor_idx;
				has_changed = true;
			}
		}
	}

	return best_index;
}

double HNSWGraph::calculate_cosine_similarity(const Embedding& vec1, const Embedding& vec2) const {
	double sum = 0;

	for (int idx = 0; idx < vec1.size(); idx++) {
		sum += vec1[idx] * vec2[idx];
	}

	return sum;
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