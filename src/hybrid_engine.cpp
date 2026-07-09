#include "../include/hybrid_engine.h"

HybridEngine::HybridEngine() {
	this->graph_index = std::make_unique<GraphIndex>();
	this->vector_index = std::make_unique<VectorIndex>();
}