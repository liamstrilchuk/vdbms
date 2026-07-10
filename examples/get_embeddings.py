from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import csv

paper_data = [line.split("\t") for line in open("data/titleabs.tsv").read().split("\n") if len(line) > 0]
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

to_run = []
to_add = []

for item in tqdm(paper_data, desc="Processing items"):
	to_run.append((item[0], item[2]))

	if len(to_run) < 64:
		continue

	vector_data = model.encode(
		[i[1] for i in to_run],
		normalize_embeddings=True,
		batch_size=64
	).tolist()

	for index, row in enumerate(vector_data):
		to_add.append((to_run[index][0], ",".join([str(round(i, 4)) for i in row])))

	to_run = []
	if len(to_add) >= 1000:
		with open("data/embeddings.csv", "a") as f:
			writer = csv.writer(f)
			writer.writerows(to_add)
		to_add = []