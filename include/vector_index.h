#pragma once
#include "types.h"

class VectorIndex final {
public:
	VectorIndex();

	void add_vector(uint32_t id, const Embedding& vec);

	uint32_t find_nearest_neighbour(const Embedding& vec) const;

private:
	std::vector<VectorNode> vectors;
};