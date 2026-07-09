#pragma once
#include "types.h"

class GraphIndex final {
public:
	GraphIndex();

	void add_edges(uint32_t source_id, const std::vector<uint32_t>& targets);

private:
	std::vector<EdgeList> adjacency_list;
};