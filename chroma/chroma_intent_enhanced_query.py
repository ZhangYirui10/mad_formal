from chroma import ChromaClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.intent_enhanced_retrieval import intent_enhanced_reformulation
import json
from tqdm import tqdm
import os

# Initialize ChromaDB client
chroma_client = ChromaClient(vector_name="evidence_bgebase", path="./chroma_store")

# Load test claims
with open("data/test.json", "r") as f:
    all_examples = json.load(f)

# Load evidence_id_to_text mapping
with open("data/evidence_id_to_text.json", "r") as f:
    evidence_id_to_text = json.load(f)

# Output file
output_file = "data/retrieved_evidence_bgebase_intent_enhanced.json"

# Load or initialize output map
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        example_to_retrieved_map = json.load(f)
    print(f"Loaded existing data with {len(example_to_retrieved_map)} examples")
else:
    example_to_retrieved_map = {}
    print("Starting fresh")

def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

for example in tqdm(all_examples, desc="Processing examples"):
    example_id = str(example["example_id"])

    if example_id in example_to_retrieved_map:
        print(f"Skipping example {example_id} - already processed")
        continue

    try:
        claim = example["claim"]

        # Step 1: Reformulate into pro and con versions
        result = intent_enhanced_reformulation(claim)
        pro_claim = result["reformulated_pro"]
        con_claim = result["reformulated_con"]

        # Step 2: Query ChromaDB with original claim, pro_claim and con_claim
        original_results = chroma_client.query_score(query_text=claim, top_k=50, include=["metadatas", "distances"])
        pro_results = chroma_client.query_score(query_text=pro_claim, top_k=50, include=["metadatas", "distances"])
        con_results = chroma_client.query_score(query_text=con_claim, top_k=50, include=["metadatas", "distances"])

        # Step 3: Process original results
        original_evidence_ids = []
        for score, meta in zip(original_results["distances"][0], original_results["metadatas"][0]):
            evidence_id = meta["evidence_id"]
            original_evidence_ids.append(evidence_id)

        # Step 4: Process pro results
        pro_evidence_ids = []
        for score, meta in zip(pro_results["distances"][0], pro_results["metadatas"][0]):
            evidence_id = meta["evidence_id"]
            pro_evidence_ids.append(evidence_id)

        # Step 5: Process con results
        con_evidence_ids = []
        for score, meta in zip(con_results["distances"][0], con_results["metadatas"][0]):
            evidence_id = meta["evidence_id"]
            con_evidence_ids.append(evidence_id)

        # Step 6: Merge results with score and evidence_id for top 20
        combined = []

        for score, meta in zip(original_results["distances"][0], original_results["metadatas"][0]):
            combined.append({
                "evidence_id": meta["evidence_id"],
                "score": score
            })

        for score, meta in zip(pro_results["distances"][0], pro_results["metadatas"][0]):
            combined.append({
                "evidence_id": meta["evidence_id"],
                "score": score
            })

        for score, meta in zip(con_results["distances"][0], con_results["metadatas"][0]):
            combined.append({
                "evidence_id": meta["evidence_id"],
                "score": score
            })

        # Step 7: Sort by score (lower is better for distances) and deduplicate evidence_ids
        seen = set()
        top_20_ids = []
        top_20_text = []
        for item in sorted(combined, key=lambda x: x["score"]):
            if item["evidence_id"] not in seen:
                top_20_ids.append(item["evidence_id"])
                top_20_text.append(evidence_id_to_text[str(item["evidence_id"])])
                seen.add(item["evidence_id"])
            if len(top_20_ids) == 20:
                break

        # Step 8: Save
        example_to_retrieved_map[example_id] = {
            "intent": result["intent"],
            "original_claim": claim,
            "pro_claim": pro_claim,
            "con_claim": con_claim,
            "original_evidence_ids": original_evidence_ids,
            "pro_evidence_ids": pro_evidence_ids,
            "con_evidence_ids": con_evidence_ids,
            "top_20_evidences_ids": top_20_ids,
            "top_20_evidences_text": top_20_text
        }

        save_to_json(example_to_retrieved_map, output_file)
        print(f"Processed example {example_id}")

    except Exception as e:
        print(f"Error on {example_id}: {e}")
        save_to_json(example_to_retrieved_map, output_file)
        continue

print(f"All done. Total processed: {len(example_to_retrieved_map)}")