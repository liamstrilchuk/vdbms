#pragma once

namespace constants {
	inline constexpr int HNSW_M = 12;
	inline constexpr int HNSW_EF_CONSTRUCTION = 96;
	inline constexpr int HNSW_EF_SEARCH = 48;
	inline constexpr int HNSW_QUERIED_IDS_MAX_SIZE = 10000;
	inline constexpr double HNSW_MIN_LAYER_PROB = 1.0e-7;
}