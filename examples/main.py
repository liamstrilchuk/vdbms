from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import requests

COUNT_PAPERS = 300
API_BASE_URL = "http://localhost:3000"

paper_data = [line.split("\t") for line in open("data/titleabs.tsv").read().split("\n")[:COUNT_PAPERS] if len(line) > 0]
paper_dict = { int(item[0]): (item[1], item[2]) for item in paper_data }

model = SentenceTransformer("all-MiniLM-L6-v2")

for item in tqdm(paper_data, desc="Processing items"):
	vector_data = model.encode(item[2], normalize_embeddings=True).tolist()
	resp = requests.post(f"{API_BASE_URL}/insert", json={
		"id": int(item[0]),
		"vector": vector_data
	})

while True:
	search_term = input("Enter search term: ")
	if not search_term:
		break

	resp = requests.post(f"{API_BASE_URL}/query", json={
		"vector": model.encode(search_term, normalize_embeddings=True).tolist()
	})
	closest = resp.json()["closest_id"]
	print(paper_dict[closest])