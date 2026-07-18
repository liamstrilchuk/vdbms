#include "../include/hnsw_graph.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <queue>
#include <immintrin.h>

HNSWGraph::HNSWGraph() : gen(std::random_device{}()), dis(0.0, 1.0) {
	this->create_layers();
}

void HNSWGraph::add_node(uint32_t node_id, Embedding& vec) {
	if (this->id_to_index.find(node_id) != this->id_to_index.end()) {
		return;
	}

	this->nodes.push_back({ node_id, 0, 0, vec, {} });
	this->id_to_index.insert({ node_id, this->nodes.size() - 1 });

	if (!this->is_initialized) {
		this->is_initialized = true;
		uint32_t last_node_index = 0;

		for (int layer = 0; layer < this->layers.size(); layer++) {
			this->layers[layer].emplace_back(HNSWNode{
				.node_index = static_cast<uint32_t>(this->nodes.size() - 1),
				.hnsw_neighbors = {-1},
				.lower_level_index = last_node_index,
				.higher_level_index = -1
			});
			if (layer > 0) {
				this->layers[layer - 1][last_node_index].higher_level_index = this->layers[layer].size() - 1;
			} else {
				this->nodes.back().base_index = this->layers[0].size() - 1;
			}
			last_node_index = this->layers[layer].size() - 1;
		}

		return;
	}

	int current_layer = this->layers.size() - 1;
	uint32_t max_add_layer = this->get_node_layer();
	int32_t last_node_index = -1;
	uint32_t entry_point = 0;

	while (current_layer > -1) {
		uint32_t closest = this->find_closest_in_layer(vec, entry_point, this->layers[current_layer]);

		if (current_layer > 0) {
			entry_point = this->layers[current_layer][closest].lower_level_index;
		}

		if (max_add_layer < current_layer) {
			current_layer--;
			continue;
		}

		auto ef_const = this->run_ef_construction(vec, closest, this->layers[current_layer], constants::HNSW_EF_CONSTRUCTION);
		auto pruned = this->prune_ef_construction(vec, ef_const, this->layers[current_layer]);

		this->layers[current_layer].emplace_back(HNSWNode{
			.node_index = static_cast<uint32_t>(this->nodes.size() - 1),
			.lower_level_index = 0,
			.higher_level_index = last_node_index
		});

		if (current_layer == 0) {
			this->nodes.back().base_index = this->layers[0].size() - 1;
		}

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
		if (idx == ef_const.size() - 1) {
			this->add_to_queried_indexes(ef_const[idx].first);
		}
		results.push_back(this->nodes[ef_const[idx].first].id);
	}

	return results;
}

void HNSWGraph::add_to_queried_indexes(uint32_t id) {
	this->last_queried_indexes.push(id);
	this->nodes[id].access_count++;

	double ratio = (double) this->nodes[id].access_count * (double) this->nodes.size() / (double) this->last_queried_indexes.size();
	if (this->nodes[id].access_count > 5 && ratio > 100) {
		uint32_t promote_to = std::floor(std::log2(ratio));
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

void HNSWGraph::create_reverse_connection(uint32_t node_id, uint32_t new_id, std::vector<HNSWNode>& layer) {
	Embedding& node_vector_data = this->nodes[layer[node_id].node_index].vector_data;
	ef_pair_list neighbor_list;
	neighbor_list.push_back({
		new_id,
		this->calculate_cosine_similarity_avx2(
			node_vector_data.data(),
			this->nodes[layer[new_id].node_index].vector_data.data(),
			384
		)
	});
	
	for (int idx = 0; idx < constants::HNSW_M; idx++) {
		if (layer[node_id].hnsw_neighbors[idx] == -1) {
			continue;
		}

		neighbor_list.push_back({
			layer[node_id].hnsw_neighbors[idx],
			this->calculate_cosine_similarity_avx2(
				node_vector_data.data(),
				this->nodes[layer[layer[node_id].hnsw_neighbors[idx]].node_index].vector_data.data(),
				384
			)
		});
	}

	std::vector<int32_t> pruned = this->prune_ef_construction(node_vector_data, neighbor_list, layer);
	std::copy(pruned.begin(), pruned.end(), layer[node_id].hnsw_neighbors);
}

ef_pair_list HNSWGraph::run_ef_construction(
	Embedding& vec, uint32_t closest, std::vector<HNSWNode>& layer, const int node_count
) const {
	this->prepare_visited_tracker(layer.size());

	struct QueueElement {
		uint32_t index;
		float score;

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

	float closest_similarity = this->calculate_cosine_similarity_avx2(
		vec.data(), this->nodes[layer[closest].node_index].vector_data.data(), 384
	);
	candidate_pool.push({ closest, closest_similarity });
	working_queue.push({ closest, closest_similarity });
	this->visited_tracker[closest] = this->current_visited_version;

	while (working_queue.size() > 0) {
		QueueElement current = working_queue.top();
		QueueElement furthest = candidate_pool.top();
		working_queue.pop();

		if (current.score < furthest.score) {
			break;
		}

		int32_t* neighbors = layer[current.index].hnsw_neighbors;

		for (int idx = 0; idx < constants::HNSW_M; idx++) {
			if (neighbors[idx] == -1 || this->visited_tracker[neighbors[idx]] == this->current_visited_version) {
				continue;
			}

			this->visited_tracker[neighbors[idx]] = this->current_visited_version;

			const Embedding& nvec = this->nodes[layer[neighbors[idx]].node_index].vector_data;
			float similarity = this->calculate_cosine_similarity_avx2(vec.data(), nvec.data(), 384);

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
			float sim = this->calculate_cosine_similarity_avx2(
				this->nodes[layer[selected_M[idx]].node_index].vector_data.data(),
				this->nodes[layer[top.first].node_index].vector_data.data(),
				384
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
	float best_similarity = this->calculate_cosine_similarity_avx2(
		vec.data(), this->nodes[layer[start_node].node_index].vector_data.data(), 384
	);
	uint32_t best_index = start_node;

	bool has_changed = true;
	while (has_changed) {
		this->step_count++;
		has_changed = false;
		int32_t* neighbors = layer[best_index].hnsw_neighbors;

		for (int idx = 0; idx < constants::HNSW_M; idx++) {
			uint32_t neighbor_idx = neighbors[idx];

			if (neighbors[idx] == -1) {
				continue;
			}

			float similarity = this->calculate_cosine_similarity_avx2(
				vec.data(), this->nodes[layer[neighbor_idx].node_index].vector_data.data(), 384
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

float HNSWGraph::calculate_cosine_similarity(const Embedding& vec1, const Embedding& vec2) const {
	float sum = 0;

	for (int idx = 0; idx < vec1.size(); idx++) {
		sum += vec1[idx] * vec2[idx];
	}

	return sum;
}

float HNSWGraph::calculate_cosine_similarity_avx2(const float* vec1, const float* vec2, size_t dim) const {
	size_t i = 0;
	__m256 sum_vec = _mm256_setzero_ps();

	for (; i + 7 < dim; i += 8) {
		__m256 v1 = _mm256_loadu_ps(vec1 + i);
		__m256 v2 = _mm256_loadu_ps(vec2 + i);
		sum_vec = _mm256_fmadd_ps(v1, v2, sum_vec);
	}

	float temp[8];
	_mm256_storeu_ps(temp, sum_vec);
	float scalar_sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
	for (; i < dim; i++) {
		scalar_sum += vec1[i] * vec2[i];
	}

	return scalar_sum;
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

void HNSWGraph::prepare_visited_tracker(size_t required_size) const {
	if (this->visited_tracker.size() < required_size) {
		this->visited_tracker.resize(required_size, 0);
	}

	this->current_visited_version++;

	if (this->current_visited_version == 0) {
		std::fill(this->visited_tracker.begin(), this->visited_tracker.end(), 0);
		this->current_visited_version = 1;
	}
}