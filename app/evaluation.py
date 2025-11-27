import json
from collections import defaultdict
from typing import Dict, List
from . import workflow, database, cache


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    parts = name.split()
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        return f"{first} {last}"
    return name


def check_answer_accuracy(answer_text: str, expected_strings: List[str]) -> Dict:
    if not answer_text:
        return {
            "contains_any": False,
            "contains_all": False,
            "matched_strings": [],
            "missing_strings": expected_strings,
            "match_count": 0,
            "total_expected": len(expected_strings),
            "accuracy": 0.0
        }

    answer_lower = answer_text.lower()
    matched = []
    missing = []

    for expected in expected_strings:
        expected_lower = expected.lower()
        
        if expected_lower in answer_lower:
            matched.append(expected)
        else:
            if len(expected.split()) >= 2:
                normalized_expected = normalize_name(expected)
                if normalized_expected in answer_lower:
                    matched.append(expected)
                else:
                    parts = expected_lower.split()
                    if len(parts) >= 2:
                        first = parts[0]
                        last = parts[-1]
                        if first in answer_lower and last in answer_lower:
                            matched.append(expected)
                        else:
                            missing.append(expected)
                    else:
                        missing.append(expected)
            else:
                expected_words = expected_lower.split()
                if all(word in answer_lower for word in expected_words if len(word) > 2):
                    matched.append(expected)
                else:
                    missing.append(expected)

    return {
        "contains_any": len(matched) > 0,
        "contains_all": len(matched) == len(expected_strings),
        "matched_strings": matched,
        "missing_strings": missing,
        "match_count": len(matched),
        "total_expected": len(expected_strings),
        "accuracy": len(matched) / len(expected_strings) if expected_strings else 0.0
    }


def evaluate_accuracy(
    conn,
    test_questions: List[Dict],
    text2cypher_cache: cache.LRUCache,
    prune_schema_cache: cache.LRUCache,
    use_cache: bool = False,
    use_prune_cache: bool = False,
) -> Dict:
    results = []
    total_accuracy = 0.0

    for i, test_item in enumerate(test_questions, 1):
        question = test_item["question"]
        expected_strings = test_item["expected_strings"]

        print(f"[{i}/{len(test_questions)}] Question: {question}")
        print(f"  Expected strings: {len(expected_strings)} items")

        query, context, answer = workflow.run_full_workflow(
            conn=conn,
            question=question,
            text2cypher_cache=text2cypher_cache,
            prune_schema_cache=prune_schema_cache,
            use_cache=use_cache,
            use_prune_cache=use_prune_cache,
            use_full_schema_cache=False,
            return_timings=False
        )

        answer_text = getattr(answer, "response", None) if answer else None
        if answer_text is None:
            answer_text = ""

        accuracy_result = check_answer_accuracy(answer_text, expected_strings)
        results.append({
            "question": question,
            "query": query,
            "answer": answer_text,
            "expected_strings": expected_strings,
            "accuracy_result": accuracy_result
        })

        total_accuracy += accuracy_result["accuracy"]

        print(f"  ✓ Matched: {accuracy_result['match_count']}/{accuracy_result['total_expected']} strings")
        print(f"  ✓ Accuracy: {accuracy_result['accuracy']:.2%}")
        print()

    avg_accuracy = total_accuracy / len(test_questions) if test_questions else 0.0
    contains_any_count = sum(1 for r in results if r["accuracy_result"]["contains_any"])
    contains_all_count = sum(1 for r in results if r["accuracy_result"]["contains_all"])

    return {
        "results": results,
        "average_accuracy": avg_accuracy,
        "contains_any_count": contains_any_count,
        "contains_all_count": contains_all_count,
        "total_questions": len(test_questions)
    }


def run_benchmark(
    conn,
    questions: List[str],
    text2cypher_cache: cache.LRUCache,
    prune_schema_cache: cache.LRUCache,
    use_cache: bool,
    use_prune_cache: bool,
    use_full_schema_cache: bool,
    label: str,
) -> List[Dict]:
    results = []
    for idx, question in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] {question[:60]}... ({label})")
        q_out, ctx, ans, timings = workflow.run_full_workflow(
            conn=conn,
            question=question,
            text2cypher_cache=text2cypher_cache,
            prune_schema_cache=prune_schema_cache,
            use_cache=use_cache,
            use_prune_cache=use_prune_cache,
            use_full_schema_cache=use_full_schema_cache,
            return_timings=True,
        )
        results.append({
            "question": question,
            "query": q_out,
            "answer": getattr(ans, "response", None) if ans else None,
            "timings": timings,
        })
    return results


def summarize_second_run(results: List[Dict]) -> tuple:
    stage_samples: defaultdict[str, List[float]] = defaultdict(list)
    for res in results:
        total_ms = 0.0
        for stage_name, duration in res["timings"]:
            ms = duration * 1000
            stage_samples[stage_name].append(ms)
            total_ms += ms
        stage_samples["total"].append(total_ms)
    stage_avg = {
        stage: sum(values) / len(values)
        for stage, values in stage_samples.items()
        if values
    }
    return dict(stage_samples), stage_avg


def evaluate_accuracy_comparison(
    conn,
    test_questions: List[Dict],
    text2cypher_cache: cache.LRUCache,
    prune_schema_cache: cache.LRUCache,
    use_cache: bool = False,
    use_prune_cache: bool = False,
) -> Dict:
    results_full = []
    results_simple = []
    total_accuracy_full = 0.0
    total_accuracy_simple = 0.0

    for i, test_item in enumerate(test_questions, 1):
        question = test_item["question"]
        expected_strings = test_item["expected_strings"]

        print(f"\n[{i}/{len(test_questions)}] Question: {question}")
        print(f"  Expected strings: {len(expected_strings)} items")
        print("\n" + "-" * 80)
        print("FULL MODE (with exemplars, postprocessing, refinement)")
        print("-" * 80)

        query_full, context_full, answer_full = workflow.run_full_workflow(
            conn=conn,
            question=question,
            text2cypher_cache=text2cypher_cache,
            prune_schema_cache=prune_schema_cache,
            use_cache=use_cache,
            use_prune_cache=use_prune_cache,
            use_full_schema_cache=False,
            return_timings=False
        )

        answer_text_full = getattr(answer_full, "response", None) if answer_full else None
        if answer_text_full is None:
            answer_text_full = ""

        accuracy_result_full = check_answer_accuracy(answer_text_full, expected_strings)
        results_full.append({
            "question": question,
            "query": query_full,
            "answer": answer_text_full,
            "expected_strings": expected_strings,
            "accuracy_result": accuracy_result_full
        })

        total_accuracy_full += accuracy_result_full["accuracy"]

        print(f"  ✓ Matched: {accuracy_result_full['match_count']}/{accuracy_result_full['total_expected']} strings")
        print(f"  ✓ Accuracy: {accuracy_result_full['accuracy']:.2%}")

        print("\n" + "-" * 80)
        print("SIMPLE MODE (without exemplars, postprocessing, refinement)")
        print("-" * 80)

        query_simple, context_simple, answer_simple = workflow.run_full_workflow_simple(
            conn=conn,
            question=question,
            prune_schema_cache=prune_schema_cache,
            use_prune_cache=use_prune_cache,
            use_full_schema_cache=False,
            return_timings=False
        )

        answer_text_simple = getattr(answer_simple, "response", None) if answer_simple else None
        if answer_text_simple is None:
            answer_text_simple = ""

        accuracy_result_simple = check_answer_accuracy(answer_text_simple, expected_strings)
        results_simple.append({
            "question": question,
            "query": query_simple,
            "answer": answer_text_simple,
            "expected_strings": expected_strings,
            "accuracy_result": accuracy_result_simple
        })

        total_accuracy_simple += accuracy_result_simple["accuracy"]

        print(f"  ✓ Matched: {accuracy_result_simple['match_count']}/{accuracy_result_simple['total_expected']} strings")
        print(f"  ✓ Accuracy: {accuracy_result_simple['accuracy']:.2%}")

        print("\n" + "-" * 80)
        print("COMPARISON")
        print("-" * 80)
        diff = accuracy_result_full["accuracy"] - accuracy_result_simple["accuracy"]
        print(f"  Full mode accuracy: {accuracy_result_full['accuracy']:.2%}")
        print(f"  Simple mode accuracy: {accuracy_result_simple['accuracy']:.2%}")
        print(f"  Difference: {diff:+.2%} ({'Full mode better' if diff > 0 else 'Simple mode better' if diff < 0 else 'Equal'})")
        print()

    avg_accuracy_full = total_accuracy_full / len(test_questions) if test_questions else 0.0
    avg_accuracy_simple = total_accuracy_simple / len(test_questions) if test_questions else 0.0

    contains_any_full = sum(1 for r in results_full if r["accuracy_result"]["contains_any"])
    contains_all_full = sum(1 for r in results_full if r["accuracy_result"]["contains_all"])
    contains_any_simple = sum(1 for r in results_simple if r["accuracy_result"]["contains_any"])
    contains_all_simple = sum(1 for r in results_simple if r["accuracy_result"]["contains_all"])

    print("=" * 80)
    print("ACCURACY SUMMARY - COMPARISON")
    print("=" * 80)
    print(f"Total questions: {len(test_questions)}")
    print()
    print("FULL MODE (with exemplars, postprocessing, refinement):")
    print(f"  Average accuracy: {avg_accuracy_full:.2%}")
    print(f"  Questions with at least one match: {contains_any_full}/{len(test_questions)} ({contains_any_full/len(test_questions)*100:.1f}%)")
    print(f"  Questions with all matches: {contains_all_full}/{len(test_questions)} ({contains_all_full/len(test_questions)*100:.1f}%)")
    print()
    print("SIMPLE MODE (without exemplars, postprocessing, refinement):")
    print(f"  Average accuracy: {avg_accuracy_simple:.2%}")
    print(f"  Questions with at least one match: {contains_any_simple}/{len(test_questions)} ({contains_any_simple/len(test_questions)*100:.1f}%)")
    print(f"  Questions with all matches: {contains_all_simple}/{len(test_questions)} ({contains_all_simple/len(test_questions)*100:.1f}%)")
    print()
    print("OVERALL COMPARISON:")
    avg_diff = avg_accuracy_full - avg_accuracy_simple
    print(f"  Average accuracy difference: {avg_diff:+.2%} ({'Full mode better' if avg_diff > 0 else 'Simple mode better' if avg_diff < 0 else 'Equal'})")
    print("=" * 80)
    print()

    return {
        "full_mode": {
            "results": results_full,
            "average_accuracy": avg_accuracy_full,
            "contains_any_count": contains_any_full,
            "contains_all_count": contains_all_full,
        },
        "simple_mode": {
            "results": results_simple,
            "average_accuracy": avg_accuracy_simple,
            "contains_any_count": contains_any_simple,
            "contains_all_count": contains_all_simple,
        },
        "comparison": {
            "average_accuracy_difference": avg_diff,
            "total_questions": len(test_questions)
        }
    }


def load_test_questions() -> List[Dict]:
    with open("data/test_questions.json", "r", encoding="utf-8") as test_file:
        return json.load(test_file)
