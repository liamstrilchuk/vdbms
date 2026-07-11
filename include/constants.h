#pragma once

namespace constants {
	inline constexpr int HNSW_M = 16;
	inline constexpr int HNSW_EF_CONSTRUCTION = 128;
	inline constexpr int HNSW_EF_SEARCH = 64;
	inline constexpr double HNSW_MIN_LAYER_PROB = 1.0e-7;
}