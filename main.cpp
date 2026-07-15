#include "include/lib/httplib.h"
#include "include/lib/json.hpp"
#include "include/hybrid_engine.h"
#include "include/hnsw_graph.h"
#include <iostream>
#include <utility>

using json = nlohmann::json;

int main(int argc, char** argv) {
	auto graph = std::make_unique<HNSWGraph>();
	httplib::Server server;

	server.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
		auto request_data = json::parse(req.body);
		uint32_t record_id = request_data["id"];
		Embedding record_vec = request_data["vector"];

		graph->add_node(record_id, record_vec);
		auto response_data = json::object({{ "status", 200 }});
		res.set_content(response_data.dump(), "application/json");
		std::cout << "Node created" << std::endl;
	});
	
	server.Post("/query", [&](const httplib::Request& req, httplib::Response& res) {
		auto request_data = json::parse(req.body);
		Embedding query_vec = request_data["vector"];

		std::vector<uint32_t> closest_ids = graph->search(query_vec, 5);
		auto response_data = json::object({{ "closest_id", closest_ids }});
		res.set_content(response_data.dump(), "application/json");
	});

	server.listen("0.0.0.0", 3000);
}