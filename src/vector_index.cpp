#include "../include/vector_index.h"

VectorIndex::VectorIndex() {
	
}

void VectorIndex::add_vector(uint32_t id, const Embedding& vec) {
	this->vectors.push_back({ id, vec });
}