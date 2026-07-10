#pragma once
#include <unordered_map>
#include "types.h"

class GraphIndex final {
public:
	GraphIndex();

	void add_edge(uint32_t source, uint32_t dest);

private:
	std::unordered_map<uint32_t, std::vector<uint32_t>> outbound_edges;
	std::unordered_map<uint32_t, std::vector<uint32_t>> inbound_edges;
};