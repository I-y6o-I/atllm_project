import os
import json
import time
import random
import requests
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ----------------------
# CONFIG
# ----------------------
NUM_PAPERS = 200
NUM_CLUSTERS = 15
SEED = 42
SAVE_DIR = "/home/kirill/Files/IU_Cources/atllm_project/ml/data/qasper_pdfs"
MAPPING_FILE = "/home/kirill/Files/IU_Cources/atllm_project/ml/data/id_mapping.json"


os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)


# ----------------------
# LOAD DATASET
# ----------------------
print("Loading QASPER dataset...")
ds = load_dataset("allenai/qasper", split="train")
print(f"Total papers: {len(ds)}")


# ----------------------
# QA RICHNESS CHECK
# ----------------------
def qa_richness(paper):
    """Check minimal answer variety."""
    qas = paper["qas"]
    answers = qas["answers"]

    has_extractive = False
    has_free_form = False
    has_evidence = False

    for ans_group in answers:
        for ans in ans_group["answer"]:
            if ans["extractive_spans"]:
                has_extractive = True
            if ans["free_form_answer"]:
                has_free_form = True
            if ans["evidence"]:
                has_evidence = True

    return has_extractive and has_free_form and has_evidence


print("Filtering papers with QA richness...")
filtered = [p for p in ds if qa_richness(p)]
print(f"Remaining papers after QA filtering: {len(filtered)}")

if len(filtered) < NUM_PAPERS:
    print("WARNING: Not enough QA-rich papers. Will auto-include non-rich papers.")
    filtered = list(ds)[:]  # fallback: use whole dataset


# ----------------------
# EMBED ABSTRACTS
# ----------------------
print("Embedding abstracts...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
abstracts = [p["abstract"] for p in filtered]
embeddings = model.encode(abstracts, show_progress_bar=True)


# ----------------------
# TOPIC CLUSTERING
# ----------------------
print(f"Clustering into {NUM_CLUSTERS} topic groups...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init="auto")
labels = kmeans.fit_predict(embeddings)

# Assign cluster

for i, p in enumerate(filtered):
    p["cluster"] = int(labels[i])

# Group by cluster
clusters = {c: [] for c in range(NUM_CLUSTERS)}
for p in filtered:
    clusters[p["cluster"]].append(p)


# ----------------------
# SAMPLING EXACTLY N PAPERS
# ----------------------
print(f"Selecting exactly {NUM_PAPERS} papers...")

target_per_cluster = int(np.ceil(NUM_PAPERS / NUM_CLUSTERS))
selected = []

# 1) Primary sampling: take up to target_per_cluster from each cluster
for c in range(NUM_CLUSTERS):
    pool = clusters[c]
    if len(pool) <= target_per_cluster:
        selected.extend(pool)
    else:
        selected.extend(random.sample(pool, target_per_cluster))

selected = selected[:NUM_PAPERS]  # trim if oversampled

# 2) If undersampled, fill remainder from unused QA-rich papers
if len(selected) < NUM_PAPERS:
    used_ids = set([p["id"] for p in selected])
    unused = [p for p in filtered if p["id"] not in used_ids]
    need = NUM_PAPERS - len(selected)

    selected.extend(unused[:need])

# 3) FINAL GUARANTEE: fallback to ANY papers from original dataset
if len(selected) < NUM_PAPERS:
    used_ids = set([p["id"] for p in selected])
    unused_all = [p for p in ds if p["id"] not in used_ids]
    need = NUM_PAPERS - len(selected)
    selected.extend(unused_all[:need])

# TRIM to exactly N
selected = selected[:NUM_PAPERS]

print(f"Final selected papers: {len(selected)}")


# ----------------------
# DOWNLOAD PDFs
# ----------------------
def download_arxiv_pdf(arxiv_id, filename):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(r.content)
                return True
        except:
            time.sleep(2)
    return False


print("Downloading PDFs...")
id_mapping = {}

for idx, paper in enumerate(tqdm(selected, ncols=80), start=1):
    int_id = f"{idx:03d}"
    arxiv_id = paper["id"]
    pdf_path = os.path.join(SAVE_DIR, f"{int_id}.pdf")

    success = download_arxiv_pdf(arxiv_id, pdf_path)
    id_mapping[int_id] = arxiv_id if success else None


# ----------------------
# SAVE ID MAPPING
# ----------------------
with open(MAPPING_FILE, "w") as f:
    json.dump(id_mapping, f, indent=2)

print("Done!")
print(f"PDFs saved in: {SAVE_DIR}")
print(f"Mapping saved to: {MAPPING_FILE}")
