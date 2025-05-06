from create_colbert_index import get_docs
import json

doc_ids, documents = get_docs("../autometrics/metrics/documentation/")

# Write the docs to a jsonl file for Pyserini
# {"id": {doc_id}, "contents": {document}}

with open("./collection/docs.jsonl", "w") as f:
    for doc_id, document in zip(doc_ids, documents):
        json.dump({"id": doc_id, "contents": document}, f)
        f.write("\n")

print("Documents written to ./collection/docs.jsonl")

print("Now you can run the Pyserini indexer on the collection/docs.jsonl file to create a BM25 index.")
print("For example, use the following command:")
print("""
python -m pyserini.index.lucene \\
  --collection JsonCollection \\
  --input collection/ \\
  --index bm25 \\
  --generator DefaultLuceneDocumentGenerator \\
  --threads 1 \\
  --storePositions --storeDocvectors --storeRaw""")