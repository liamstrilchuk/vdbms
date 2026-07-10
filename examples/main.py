from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import requests

COUNT_PAPERS = 5000
API_BASE_URL = "http://localhost:3000"

paper_data = [line.split("\t") for line in open("data/titleabs.tsv").read().split("\n")[:COUNT_PAPERS] if len(line) > 0]
paper_dict = { int(item[0]): (item[1], item[2]) for item in paper_data }

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

model = SentenceTransformer("all-MiniLM-L6-v2")

vector_data_by_pid = {}

for item in tqdm(paper_data, desc="Processing items"):
	vector_data = model.encode(item[2], normalize_embeddings=True).tolist()
	vector_data_by_pid[int(item[0])] = vector_data

for item in paper_data:
	if not int(item[0]) in edge_map_by_origin:
		all_neighbours = []
	else:
		all_neighbours = edge_map_by_origin[int(item[0])]
	
	included_neighbours = [n for n in all_neighbours if n in vector_data_by_pid]

	resp = requests.post(f"{API_BASE_URL}/insert", json={
		"id": int(item[0]),
		"vector": vector_data,
		"edges": included_neighbours
	})

# while True:
# 	search_term = input("Enter search term: ")
# 	if not search_term:
# 		break

# 	resp = requests.post(f"{API_BASE_URL}/query", json={
# 		"vector": model.encode(search_term, normalize_embeddings=True).tolist()
# 	})
# 	closest = resp.json()["closest_id"]
# 	print(paper_dict[closest])