"""RAG retrieval evaluation runner — NOT a pytest test (no test_ prefix,
not auto-collected). Unlike tests/unit/*, this hits the real embedding
provider, vector store, and search strategy configured in app.config;
it measures actual retrieval quality against whatever is currently
ingested, rather than proving a plug-in's contract with fakes.

Fixtures: one *.json file per RAG domain in this directory, each a list
of {"question": str, "tags": dict, "expected_doc_id": str}. The domain
name is the filename stem (e.g. compliance.json -> domain "compliance").

Run: uv run python -m tests.rag_eval.run_eval
"""
import asyncio
import json
from pathlib import Path
from typing import Any

from app.rag import registry as rag_registry

_FIXTURES_DIR = Path(__file__).parent
_TOP_K = 5


def _load_fixtures() -> dict[str, list[dict[str, Any]]]:
    fixtures: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(_FIXTURES_DIR.glob("*.json")):
        fixtures[path.stem] = json.loads(path.read_text())
    return fixtures


async def _eval_one(domain: str, case: dict[str, Any]) -> dict[str, Any]:
    question = case["question"]
    tags = case.get("tags") or {}
    expected = case["expected_doc_id"]

    results = await rag_registry.get(domain).search(question, filters=tags, top_k=_TOP_K)
    doc_ids = [r["doc_id"] for r in results]

    rank = doc_ids.index(expected) + 1 if expected in doc_ids else None
    return {
        "domain": domain,
        "question": question,
        "expected": expected,
        "rank": rank,
        "hit": rank is not None,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        "top_result": doc_ids[0] if doc_ids else None,
    }


async def main() -> None:
    fixtures = _load_fixtures()
    if not fixtures:
        print(f"No fixture files found in {_FIXTURES_DIR}")
        return

    all_results: list[dict[str, Any]] = []
    for domain, cases in fixtures.items():
        for case in cases:
            all_results.append(await _eval_one(domain, case))

    print(f"\n{'domain':<12} {'hit@' + str(_TOP_K):<8} {'rank':<6} question")
    print("-" * 90)
    for r in all_results:
        status = f"HIT@{r['rank']}" if r["hit"] else "MISS"
        marker = "OK  " if r["hit"] else "FAIL"
        print(f"{marker} {r['domain']:<12} {status:<8} {r['question'][:55]}")
        if not r["hit"]:
            print(f"       expected: {r['expected']}")
            print(f"       got:      {r['top_result'] or '(no results)'}")

    print("\n--- summary by domain ---")
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for r in all_results:
        by_domain.setdefault(r["domain"], []).append(r)

    for domain, results in sorted(by_domain.items()):
        n = len(results)
        hits = sum(1 for r in results if r["hit"])
        mrr = sum(r["reciprocal_rank"] for r in results) / n
        print(f"{domain:<14} hit@{_TOP_K}: {hits}/{n} ({hits / n:.0%})   MRR: {mrr:.2f}")

    total = len(all_results)
    total_hits = sum(1 for r in all_results if r["hit"])
    overall_mrr = sum(r["reciprocal_rank"] for r in all_results) / total if total else 0.0
    print("-" * 40)
    print(f"{'overall':<14} hit@{_TOP_K}: {total_hits}/{total} ({total_hits / total:.0%})   MRR: {overall_mrr:.2f}" if total else "no fixtures found")


if __name__ == "__main__":
    asyncio.run(main())
