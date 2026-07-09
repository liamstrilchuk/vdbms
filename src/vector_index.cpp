#include "../include/vector_index.h"

VectorIndex::VectorIndex() {
	
}

void VectorIndex::add_vector(uint32_t id, const Embedding& vec) {
	this->vectors.push_back({ id, vec });
}

uint32_t VectorIndex::find_nearest_neighbour(const Embedding& vec) const {
	if (this->vectors.size() == 0) {
		return 0;
	}

	double closest_value = -2;
	uint32_t closest_id = 0;

	for (auto vector : this->vectors) {
		double sum = 0;

		for (int idx = 0; idx < vec.size(); idx++) {
			sum += vec[idx] * vector.vector_data[idx];
		}

		if (sum > closest_value) {
			closest_value = sum;
			closest_id = vector.id;
		}
	}

	return closest_id;
}