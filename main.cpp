#include "include/lib/httplib.h"
#include "include/lib/json.hpp"
#include "include/hybrid_engine.h"
#include <iostream>

using json = nlohmann::json;

int main(int argc, char** argv) {
	auto engine = std::make_unique<HybridEngine>();
	httplib::Server server;

	server.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
		auto request_data = json::parse(req.body);
		uint32_t record_id = request_data["id"];
		Embedding record_vec = request_data["vector"];
		std::vector<uint32_t> edges = request_data["edges"];

		engine->insert_record(record_id, record_vec, edges);
		auto response_data = json::object({{ "status", 200 }});
		res.set_content(response_data.dump(), "application/json");
	});

	server.Post("/query", [&](const httplib::Request& req, httplib::Response& res) {
		auto request_data = json::parse(req.body);
		Embedding query_vec = request_data["vector"];

		uint32_t closest_id = engine->find_nearest_neighbour(query_vec);
		auto response_data = json::object({{ "closest_id", closest_id }});
		res.set_content(response_data.dump(), "application/json");
	});

	server.listen("0.0.0.0", 3000);
}