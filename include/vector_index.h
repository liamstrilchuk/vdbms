#pragma once
#include "types.h"

class VectorIndex final {
public:
	VectorIndex();

	void add_vector(uint32_t id, const Embedding& vec);

private:
	std::vector<VectorNode> vectors;
};