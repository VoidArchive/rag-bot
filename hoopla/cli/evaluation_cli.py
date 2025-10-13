import argparse
import json
import os

from lib.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # Load the golden dataset
    golden_dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "golden_dataset.json")
    with open(golden_dataset_path, 'r') as f:
        golden_data = json.load(f)

    print(f"k={limit}")
    print()

    # Evaluate each test case
    for test_case in golden_data["test_cases"]:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])

        # Run RRF search with k=60 and the specified limit
        result = rrf_search_command(query, k=60, limit=limit)
        retrieved_titles = [res["title"] for res in result["results"]]

        # Calculate precision and recall
        retrieved_set = set(retrieved_titles)
        relevant_retrieved = retrieved_set.intersection(relevant_docs)
        precision = len(relevant_retrieved) / len(retrieved_titles) if retrieved_titles else 0
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

        # Print results
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(sorted(relevant_docs))}")
        print()


if __name__ == "__main__":
    main()
