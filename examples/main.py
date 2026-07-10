import requests
import csv

COUNT_PAPERS = 5000
API_BASE_URL = "http://localhost:3000"

paper_data = [line.split("\t") for line in open("data/titleabs.tsv").read().split("\n")[:COUNT_PAPERS] if len(line) > 0]
paper_dict = { int(item[0]): [item[1], item[2], None] for item in paper_data }

split_and_cast = lambda data: [[int(x) for x in line.split(",")] for line in data if len(line) > 0]

mapping_lines = open("data/nodeidx2paperid.csv").read().split("\n")[1:]
mapping_data = split_and_cast(mapping_lines)
mapping_dict_pid_nid = { node[1]: node[0] for node in mapping_data }
mapping_dict_nid_pid = { node[0]: node[1] for node in mapping_data }

edge_lines = open("data/edge.csv").read().split("\n")
edge_data = split_and_cast(edge_lines)
edge_map_by_origin = {}

for edge in edge_data:
	if not mapping_dict_nid_pid[edge[0]] in edge_map_by_origin:
		edge_map_by_origin[mapping_dict_nid_pid[edge[0]]] = []

	edge_map_by_origin[mapping_dict_nid_pid[edge[0]]].append(mapping_dict_nid_pid[edge[1]])

with open("data/embeddings.csv", "r") as f:
	reader = csv.reader(f)
	for line in reader:
		if int(line[0]) in paper_dict:
			paper_dict[int(line[0])][2] = [float(x) for x in line[1].split(",")]

all_neighbour_pairs = []

for item in paper_data:
	pid = int(item[0])
	all_neighbours = edge_map_by_origin[pid] if pid in edge_map_by_origin else []
	included_neighbours = [n for n in all_neighbours if n in paper_dict]
	all_neighbour_pairs.extend([[pid, nb] for nb in included_neighbours])
	
	vector_data = paper_dict[int(item[0])][2]

	if vector_data is None:
		print(item[0], "has no vector data")

	resp = requests.post(f"{API_BASE_URL}/insert", json={
		"id": int(item[0]),
		"vector": vector_data
	})

resp = requests.post(f"{API_BASE_URL}/create-edges", json={
	"edges": all_neighbour_pairs
})
print(resp.json())