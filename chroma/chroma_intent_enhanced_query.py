from chroma import ChromaClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.intent_enhanced_retrieval import intent_enhanced_reformulation
import json
from tqdm import tqdm
import os

# Fix paths - use relative paths from the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Initialize ChromaDB client
chroma_client = ChromaClient(vector_name="evidence_bgebase", path="./chroma_store")

# Load test claims with correct path
test_file_path = os.path.join(project_root, "data", "test.json")
with open(test_file_path, "r") as f:
    all_examples = json.load(f)
print(f"Loaded {len(all_examples)} examples from test.json")

# Load evidence_id_to_text mapping with correct path
evidence_file_path = os.path.join(project_root, "data", "evidence_id_to_text.json")
with open(evidence_file_path, "r") as f:
    evidence_id_to_text = json.load(f)
print(f"Loaded {len(evidence_id_to_text)} evidence mappings")

# Output file
output_file = os.path.join(project_root, "data", "retrieved_evidence_bgebase_intent_enhanced.json")

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

# Test ChromaDB connection first
print("Testing ChromaDB connection...")
try:
    test_results = chroma_client.query_score(query_text="test", top_k=1, include=["metadatas", "distances"])
    print("ChromaDB connection successful.")
except Exception as e:
    print(f"ChromaDB connection failed: {e}")
    sys.exit(1)

# Test model loading
print("Testing model loading...")
try:
    test_claim = "This is a test claim."
    test_result = intent_enhanced_reformulation(test_claim)
    print("Model loading successful.")
except Exception as e:
    print(f"Model loading failed: {e}")
    sys.exit(1)

for example in tqdm(all_examples, desc="Processing examples"):
    example_id = str(example["example_id"])

    if example_id in example_to_retrieved_map:
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

        # Step 3: Process pro results
        pro_evidence_ids = []
        for score, meta in zip(pro_results["distances"][0], pro_results["metadatas"][0]):
            evidence_id = meta["evidence_id"]
            pro_evidence_ids.append(evidence_id)

        # Step 4: Process con results
        con_evidence_ids = []
        for score, meta in zip(con_results["distances"][0], con_results["metadatas"][0]):
            evidence_id = meta["evidence_id"]
            con_evidence_ids.append(evidence_id)

        # Step 5: Merge results with score and evidence_id for top 20
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

        # Step 6: Sort by score (lower is better for distances) and deduplicate evidence_ids
        seen = set()
        top_20_ids = []
        top_20_text = []
        for item in sorted(combined, key=lambda x: x["score"]):
            if item["evidence_id"] not in seen:
                top_20_ids.append(item["evidence_id"])
                evidence_id_str = str(item["evidence_id"])
                if evidence_id_str in evidence_id_to_text:
                    top_20_text.append(evidence_id_to_text[evidence_id_str])
                else:
                    top_20_text.append("Evidence not found")
                seen.add(item["evidence_id"])
            if len(top_20_ids) == 20:
                break

        # Step 7: Save
        example_to_retrieved_map[example_id] = {
            "claim": claim,
            "intent": result["intent"],
            "pro_claim": pro_claim,
            "con_claim": con_claim,
            "pro_evidence_ids": pro_evidence_ids,
            "con_evidence_ids": con_evidence_ids,
            "top_20_evidences_ids": top_20_ids,
            "evidence_full_text": top_20_text
        }

        save_to_json(example_to_retrieved_map, output_file)

    except Exception as e:
        print(f"Error on {example_id}: {e}")
        import traceback
        traceback.print_exc()
        save_to_json(example_to_retrieved_map, output_file)
        continue

print(f"All done. Total processed: {len(example_to_retrieved_map)}")
print(f"Output saved to: {output_file}")