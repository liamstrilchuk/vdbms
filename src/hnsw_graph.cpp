#include "../include/hnsw_graph.h"
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

	while (current_layer > -1) {
		uint32_t closest = this->find_closest_in_layer(vec, 0, this->layers[current_layer]);
		if (max_add_layer >= current_layer) {
			std::vector<uint32_t> ef_const = this->run_ef_construction(vec, closest, this->layers[current_layer]);
		}
		current_layer--;
	}
}

std::vector<uint32_t> HNSWGraph::run_ef_construction(Embedding& vec, uint32_t closest, std::vector<HNSWNode>& layer) const {
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
	std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>> candidate_pool;

	double closest_similarity = this->calculate_cosine_similarity(vec, this->nodes[layer[closest].node_index].vector_data);
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
				bool should_add = candidate_pool.size() < constants::HNSW_EF_CONSTRUCTION;

				if (candidate_pool.size() >= constants::HNSW_EF_CONSTRUCTION && similarity > worst.score) {
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

	std::vector<uint32_t> result;

	while (!candidate_pool.empty()) {
		result.push_back(candidate_pool.top().index);
		candidate_pool.pop();
	}

	return result;
}

uint32_t HNSWGraph::find_closest_in_layer(Embedding& vec, uint32_t start_node, std::vector<HNSWNode>& layer) const {
	double best_similarity = this->calculate_cosine_similarity(vec, this->nodes[layer[start_node].node_index].vector_data);
	HNSWNode* best_node = &layer[start_node];
	uint32_t best_index = start_node;
	std::unordered_set<uint32_t> explored_nodes;
	std::vector<uint32_t> priority_queue;

	for (int idx = 0; idx < constants::HNSW_M; idx++) {
		if (best_node->hnsw_neighbors[idx] != -1) {
			priority_queue.push_back(best_node->hnsw_neighbors[idx]);
			explored_nodes.insert(best_node->hnsw_neighbors[idx]);
		}
	}

	while (priority_queue.size() > 0) {
		uint32_t current_idx = priority_queue.back();
		HNSWNode* current = &layer[current_idx];
		priority_queue.pop_back();

		double similarity = this->calculate_cosine_similarity(vec, this->nodes[current->node_index].vector_data);

		if (similarity < best_similarity) {
			continue;
		}

		best_similarity = similarity;
		best_node = current;
		best_index = current_idx;

		for (int idx = 0; idx < constants::HNSW_M; idx++) {
			auto val = current->hnsw_neighbors[idx];
			if (val != -1 && explored_nodes.find(val) == explored_nodes.end()) {
				priority_queue.push_back(val);
				explored_nodes.insert(val);
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