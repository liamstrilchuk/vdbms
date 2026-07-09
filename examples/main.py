from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import requests

COUNT_PAPERS = 300

paper_data = [line.split("\t") for line in open("data/titleabs.tsv").read().split("\n")[:COUNT_PAPERS] if len(line) > 0]
paper_dict = { item[0]: (item[1], item[2]) for item in paper_data }

model = SentenceTransformer("all-MiniLM-L6-v2")

for item in tqdm(paper_data, desc="Processing items"):
	vector_data = model.encode(item[2]).tolist()
	resp = requests.post("http://localhost:3000/insert", json={
		"id": int(item[0]),
		"vector": vector_data
	})