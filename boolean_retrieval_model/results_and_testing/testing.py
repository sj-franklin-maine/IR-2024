from ranx import Qrels, Run, evaluate
import pandas as pd

qrel_file = "qrel_1.tsv"
run_file = "result_binary_1.tsv"

# Read the QREL file
qrels = Qrels.from_file(qrel_file, kind="trec")
run = Run.from_file(run_file, kind="trec")

# Evaluate metrics
metrics = ["precision@1", "precision@5", "ndcg@5", "mrr", "map"]
results = evaluate(qrels, run, metrics)

for metric, score in results.items():
    print(f"{metric}: {score}")
