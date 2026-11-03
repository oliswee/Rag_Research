"""Publish sanitized measured results with explicit incomplete human acceptance."""
# Markdown paragraph literals remain intact for the generated document.
# ruff: noqa: E501

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.final_contracts import (
    load_jsonl,
    sha256,
    validate_annotation,
    verify_freeze,
    verify_models,
    verify_runtime,
    write_jsonl,
)


def csv_file(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields or list(rows[0]), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def protect_human_artifacts(destination):
    gold = destination / "gold_annotations.jsonl"
    if gold.exists() and gold.stat().st_size:
        raise ValueError("Preserve existing Gold annotations")
    pdf = destination / "pdf_parser_human_audit.csv"
    if pdf.exists():
        with pdf.open(encoding="utf-8", newline="") as stream:
            if any(
                row.get("reviewer_id") or row.get("verdict")
                for row in csv.DictReader(stream)
            ):
                raise ValueError("Preserve completed PDF review decisions")


def run(root, destination):
    controlled_name = (
        "controlled_generation_fixed"
        if (root / "controlled_generation_fixed").exists()
        else "controlled_generation"
    )
    required = [
        "retrieval_replay/summary.json",
        "generation/summary.json",
        controlled_name + "/summary.json",
        controlled_name + "/original_sources.db",
        "pdf_audit/current_review_freeze.json",
        "stress.json",
        "web_routes.json",
        "ai_error_review.jsonl",
        "model_freeze.json",
        "raw_long_context_pairs.jsonl",
        "actual_web_flow_verified.json",
    ]
    if any(not (root / name).is_file() for name in required):
        raise ValueError("Finish measured runs before publishing a final report")

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    frozen = read("freeze_config.json")
    verify_freeze(root, frozen)
    verify_runtime(frozen)
    model_verification = verify_models(root, frozen)
    retrieval = read("retrieval_replay/summary.json")
    rows = load_jsonl(root / "retrieval_replay/case_results.jsonl")
    generated = load_jsonl(root / "generation/results.jsonl")
    controlled = load_jsonl(root / controlled_name / "results.jsonl")
    documents = {r["paper_id"]: r["text"] for r in load_jsonl(root / "papers.jsonl")}
    human = []
    for annotation in load_jsonl(root / "gold_annotations.jsonl"):
        try:
            validate_annotation(annotation, documents, require_human=True)
        except ValueError:
            continue
        human.append(annotation)
    destination.mkdir(parents=True, exist_ok=True)
    protect_human_artifacts(destination)
    if human:
        raise ValueError(
            "Review human annotation publication explicitly before exporting quotes"
        )
    # Empty file is intentional: pending dataset references are not published Gold.
    (destination / "gold_annotations.jsonl").write_text("", encoding="utf-8")
    (destination / "ablation.csv").write_bytes(
        (root / "retrieval_replay/ablation.csv").read_bytes()
    )
    with (root / "pdf_audit/current_parser_human_audit.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        pdf_forms = list(csv.DictReader(stream))
    csv_file(destination / "pdf_parser_human_audit.csv", pdf_forms)
    published = []
    citation_rows = []
    for scope, items in (
        ("public_reference_development", generated),
        ("synthetic_controlled", controlled),
    ):
        for row in items:
            for citation in row["answer"].get("citations", []):
                citation_rows.append(
                    {
                        "scope": scope,
                        "case_id": row["case_id"],
                        "source_origin": citation["source_origin"],
                        "evidence_type": citation["evidence_type"],
                        "source_id": citation["source_id"],
                        "doi_present": bool(citation.get("doi")),
                        "doi_registration_verified": None,
                        "pdf_page": citation.get("pdf_page"),
                        "raw_quote_sha256": sha256(citation["raw_quote"].encode()),
                        "canonical_binding_pass": row["deterministic"][
                            "citation_integrity"
                        ],
                        "human_claim_support": None,
                    }
                )
            published.append(
                {
                    "scope": scope,
                    **{
                        key: row.get(key)
                        for key in (
                            "case_id",
                            "provenance",
                            "support_status",
                            "pipeline_llm_calls",
                            "pipeline_token_usage",
                            "pipeline_budget",
                            "stage_ms",
                            "pipeline_failures",
                            "deterministic",
                            "metric_target",
                            "human_review_status",
                            "metrics",
                            "acquisition",
                            "answerable_reference",
                        )
                    },
                }
            )
    write_jsonl(destination / "deepeval_final_results.jsonl", published)
    csv_file(
        destination / "citation_checks.csv",
        citation_rows,
        fields=[
            "scope",
            "case_id",
            "source_origin",
            "evidence_type",
            "source_id",
            "doi_present",
            "doi_registration_verified",
            "pdf_page",
            "raw_quote_sha256",
            "canonical_binding_pass",
            "human_claim_support",
        ],
    )
    csv_file(
        destination / "reference_abstention.csv",
        [
            {
                "case_id": row["case_id"],
                "answerable_dataset_reference": row["answerable_reference"],
                "actual_support_status": row["support_status"],
                "generated_supported_claims": row["deterministic"][
                    "supported_claim_count"
                ],
                "human_correct_refusal": None,
                "human_false_refusal": None,
            }
            for row in generated
        ],
    )
    metric_groups = defaultdict(list)
    for row in published:
        for metric in row["metrics"]:
            metric_groups[
                (row["scope"], row["metric_target"], metric["metric"])
            ].append(metric)
    metric_table = []
    for (scope, target, name), metrics in sorted(metric_groups.items()):
        scores = [
            m["score"]
            for m in metrics
            if m["status"] == "evaluated"
            and type(m.get("score")) in {int, float}
            and math.isfinite(m["score"])
            and 0 <= m["score"] <= 1
        ]
        metric_table.append(
            {
                "scope": scope,
                "metric_target": target,
                "metric": name,
                "requested": len(metrics),
                "scored": len(scores),
                "mean": sum(scores) / len(scores) if scores else None,
                "not_evaluated": sum(m["status"] == "not_evaluated" for m in metrics),
                "not_applicable": sum(m["status"] == "not_applicable" for m in metrics),
            }
        )
    csv_file(destination / "deepeval_summary.csv", metric_table)
    csv_file(
        destination / "long_context.csv",
        load_jsonl(root / "generation/long_context_eligibility.jsonl"),
    )
    long_pairs = load_jsonl(root / "raw_long_context_pairs.jsonl")
    csv_file(
        destination / "long_context_comparison.csv",
        [
            {
                "case_id": row["case_id"],
                "mode": row.get("mode"),
                "status": row["status"],
                "context_budget": row.get("context_budget"),
                "support_status": row.get("result", {}).get("support_status"),
                "pipeline_calls": row.get("result", {}).get("pipeline_llm_calls"),
                "human_answer_completeness": None,
                "judge_comparison": "not_evaluated",
            }
            for row in long_pairs
        ],
    )
    order = load_jsonl(root / controlled_name / "evidence_order.jsonl")
    csv_file(
        destination / "evidence_order.csv",
        [
            {
                "order": r["order"],
                "same_evidence_set": r.get("same_evidence_set"),
                "support_status": r.get("result", {}).get("support_status"),
                "pipeline_calls": r.get("result", {}).get("pipeline_llm_calls"),
                "human_completeness": None,
            }
            for r in order
        ],
    )
    web = read("web_routes.json")
    csv_file(
        destination / "web_fallback.csv",
        [
            {
                "scope": "injected_route_fixture",
                "case": row["case"],
                "expected_web": row["expected_web"],
                "actual_web": row["actual_web"],
                "human_reviewed": False,
            }
            for row in web["routes"]
        ],
    )
    actual_web = read("actual_web_flow_verified.json")
    csv_file(
        destination / "actual_web_flow.csv",
        [
            {
                key: row.get(key)
                for key in (
                    "scope",
                    "run_id",
                    "local_only",
                    "support_status",
                    "citation_count",
                    "evidence_count",
                    "web_record_count",
                    "external_attempts",
                    "pipeline_calls",
                    "human_correctness",
                )
            }
            for row in actual_web
        ],
    )
    write_jsonl(
        destination / "paired_results.jsonl",
        [
            {"variant": name, "metrics": value}
            for name, value in retrieval["paired"].items()
        ],
    )
    csv_file(
        destination / "per_case.csv",
        rows,
        fields=[
            "case_id",
            "variant",
            "answerable_reference",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "prompt_evidence_coverage",
            "visible_context_evidence_coverage",
            "reference_context_precision",
            "candidate_count",
            "packed_count",
            "funnel",
            "retrieval_ms",
            "retrieval_timing_shared_with",
        ],
    )
    reviews = load_jsonl(root / "ai_error_review.jsonl")
    error_text = "# Evidence-loss and reference-label review\n\nThese ten cases were read by Codex against the reference paragraph and actual context excerpt. Reviewer type is model-assisted, not human. Character coverage of a coarse dataset paragraph is not question-answer completeness. Labels remain pending independent source review.\n\n"
    for review in reviews:
        case = next(
            r
            for r in rows
            if r["case_id"] == review["case_id"]
            and r["variant"] == "multi_rerank_parent"
        )
        error_text += f"- `{review['case_id']}` — {case['funnel']}; citable coverage={case['prompt_evidence_coverage']:.4f}. {review['assessment']}\n"
    error_text += "\nThe public references contain BIBREF placeholders, broad/compound questions and evidence paragraphs longer than minimal answers. These are explicit annotation risks, not new human Gold. Generation/Judge disagreements and absent human claim-support labels remain in the result JSONL.\n"
    (destination / "error_analysis.md").write_text(error_text, encoding="utf-8")
    evaluator = {
        p.name: sha256(p.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
        for p in sorted(Path(__file__).parent.glob("final*.py"))
    }
    freeze = {
        **frozen,
        "evaluation_source_sha256": evaluator,
        "retrieval_replay_input": read("retrieval_replay/replay_input_freeze.json"),
        "pdf_current_review_freeze": read("pdf_audit/current_review_freeze.json"),
        "formal_human_gold": 0,
        "actual_cached_models": read("model_freeze.json"),
        "model_identity_verification": model_verification,
        "run_date": "2026-11-04",
    }
    (destination / "freeze_config.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8"
    )
    deterministic = [r["deterministic"] for r in generated + controlled]
    gate = {
        "formal_m8_accepted": False,
        "models_in_initial_prospective_freeze": model_verification[
            "models_in_initial_prospective_freeze"
        ],
        "independent_human_gold": len(human),
        "independent_human_pdf_reviews": 0,
        "independent_human_error_reviews": 0,
        "human_claim_support": None,
        "real_user_study_participants": 0,
        "fake_pdf_pages": sum(r["fake_pdf_pages"] for r in deterministic),
        "local_only_web_citations": sum(
            r["local_only_web_citations"] for r in deterministic
        ),
        "citation_binding_failures": sum(
            not r["citation_integrity"] for r in deterministic
        ),
        "credential_gate": "existing_M7_and_final_security_tests_must_pass",
        "new_default_enabled": False,
        "new_ui_screenshot": "blocked_by_automatic_approval",
        "pending": [
            "30–50 independently verified human Gold cases",
            "50 current-parser human PDF reviews",
            "10 human error-attribution checks",
            "human claim-support calibration for six scenario classes",
        ],
        "optional_user_study": "not_performed_no_participants_available",
    }
    (destination / "release_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    cost_rounds = []
    for label in ("generation", "controlled_generation", "controlled_generation_fixed"):
        source = root / label / "results.jsonl"
        if source.exists():
            observed = load_jsonl(source)
            cost_rounds.append(
                {
                    "round": label,
                    "cases": len(observed),
                    "pipeline_successful_calls": sum(
                        r["pipeline_llm_calls"] for r in observed
                    ),
                    "pipeline_model_attempts": sum(
                        r["pipeline_budget"]["model_attempts"] for r in observed
                    ),
                    "judge_attempts": sum(
                        m["judge_attempts"] for r in observed for m in r["metrics"]
                    ),
                    "counted_in_final_scenario_table": label
                    in {"generation", controlled_name},
                }
            )
        ordering = root / label / "evidence_order.jsonl"
        if ordering.exists():
            observed = load_jsonl(ordering)
            cost_rounds.append(
                {
                    "round": label + "/ordering",
                    "cases": len(observed),
                    "pipeline_successful_calls": sum(
                        r.get("result", {}).get("pipeline_llm_calls", 0)
                        for r in observed
                    ),
                    "judge_attempts": 0,
                    "counted_in_final_scenario_table": False,
                }
            )
    for label, observed in (
        (
            "initial_all_native_children",
            load_jsonl(root / "generation/long_context_results.jsonl"),
        ),
        (
            "matched_raw_full_paper_pairs",
            [r["result"] for r in long_pairs if "result" in r],
        ),
    ):
        cost_rounds.append(
            {
                "round": label,
                "cases": len(observed),
                "pipeline_successful_calls": sum(
                    r["pipeline_llm_calls"] for r in observed
                ),
                "pipeline_model_attempts": sum(
                    r["pipeline_budget"]["model_attempts"] for r in observed
                ),
                "judge_attempts": 0,
                "counted_in_final_scenario_table": False,
            }
        )
    cost_rounds.append(
        {
            "round": "guarded_actual_web_flow",
            "cases": len(actual_web),
            "pipeline_successful_calls": sum(r["pipeline_calls"] for r in actual_web),
            "external_attempts": sum(r["external_attempts"] for r in actual_web),
            "judge_attempts": 0,
            "counted_in_final_scenario_table": False,
        }
    )
    if (root / "demo_observations.json").exists():
        demo_cost = read("demo_observations.json")
        cost_rounds.append(
            {
                "round": "isolated_API_walkthrough",
                "pipeline_successful_calls": demo_cost["observed_pipeline_calls"],
                "pipeline_model_attempts": demo_cost["observed_model_attempts"],
                "judge_attempts": 0,
                "counted_in_final_scenario_table": False,
            }
        )
    cost = {
        "rounds": cost_rounds,
        "initial_rewrite_calls": 39,
        "replay_rewrite_calls": 0,
        "provider_api_currency": 0,
        "total_local_resource_cost": None,
        "initial_unguarded_web_probe": "Windows spawn re-executed helper; duplicated external/model work and mixed trace pairing, total usage unknown. Original log retained, excluded from paired quality results.",
        "connectivity_probe_external_requests": 2,
        "notes": "Includes initial fixture failure, corrected rerun, ordering, both long-context rounds, verified Web probe and API demo. Failed/duplicated work has unknown total usage; successful responses/tokens are lower bounds, not total resource cost.",
    }
    (destination / "diagnostic_cost_ledger.json").write_text(json.dumps(cost, indent=2))
    report = "# M8 final development evaluation and delivery\n\nDate:2026-11-04 (Australia/Sydney). M7 PR60 merged as`2b48206285f3237ce7590b3e5593c415bd199bda`; M1–M7 production source/config frozen for this run. M8 adds evaluation/delivery tooling. **Independent human M8 acceptance is not complete.** The engineering results below use public dataset references, explicitly controlled source fixtures and separately labelled AI review. No overall scientific-answer accuracy is certified.\n\n"
    report += "## Corpus and methods\n\n40 new QASPER train papers/questions,32 answerable/8 unanswerable dataset references; earlier dev40 excluded.929395 text characters map to849 native children. Native provided-text spans/pages-null are checked against exact raw offsets. Dataset labels, model pretraining exposure and scientific entailment were not independently certified. Paper Hit is conditional on a supplied paper scope, not whole-library paper discovery.\n\nActual local SQLite FTS, BGE-small-en-v1.5(384), Qdrant1.16.2 and cached cross-encoder were used. Seven variants share corpus/models/budgets. The initial run made39 actual rewrite calls:13 plans applied,26 rejected to Q0 and1 disabled by constraint guard. Final replay preserves those exact plans, reruns actual retrieval/reranking with no new rewrite calls, and records complete R0/ranking diagnostics. Initial artifacts remain intact.\n\nPacking is scored separately from rank lists. The table below uses admitted child Recall@10 after packing. Full original-paragraph character coverage is a reference proxy, not answer completeness; visible parent background is not automatically citable child evidence. Missing/unanswerable metrics remain null. Retrieval timings exclude rewrite/model-generation cost; derived packing variants share the same measured retrieval. Some diagnostic jobs overlapped, so timings are workstation observations, not an isolated SLA.\n\n"
    report += "| Variant | Packed Recall@10 | MRR | Citable coverage | Visible coverage | Retrieval p50/p95 ms |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    for name, values in retrieval["variants"].items():
        report += f"| {name} | {values['recall_at_10']:.6f} | {values['mrr_at_10']:.6f} | {values['prompt_evidence_coverage']:.6f} | {values['visible_context_evidence_coverage']:.6f} | {values['retrieval_p50_ms']:.2f}/{values['retrieval_p95_ms']:.2f} |\n"
    paired = retrieval["paired"]["multi_rerank_parent"]["recall_at_10"]
    report += f"\nCompared with single Hybrid flat, final packed recall mean delta={paired['mean_delta']:.6f}; paired wins/ties/losses={paired['win_tie_loss']}; percentile bootstrap95% interval={paired['bootstrap_95_percentile']} ({paired['n']} answerable pairs,2000 draws,seed61). This is descriptive reference development, not a confirmatory significance/quality claim. See paired_results.jsonl and per_case.csv. No production default was changed to fit these questions.\n"
    report += "\n## Generation, citations and Judge\n\nEight actual public-reference generations and six controlled local-PDF/Web-OA/abstract/metadata/fault/conflict scenarios are recorded separately. Synthetic Web originals are injected licensed fixtures, not actual Web retrieval gains. Declared metadata/fault cases have no factual evidence. Claim support is verified by the production pipeline but remains uncalibrated against humans. Exact citation ID/quote/DOI-field/URL/page binding is independently checked; DOI-field equality is not verification that a DOI is registered or that a publication is scientifically true.\n\n"
    report += f"Public-reference statuses:{dict(Counter(r['support_status'] for r in generated))}; controlled statuses:{dict(Counter(r['support_status'] for r in controlled))}. Public pipeline calls={sum(r['pipeline_llm_calls'] for r in generated)}; controlled pipeline calls={sum(r['pipeline_llm_calls'] for r in controlled)}. Judge attempts={sum(m['judge_attempts'] for r in generated + controlled for m in r['metrics'])}, separately from pipeline and rewrite calls. Detailed token counts, failed/unknown usage, metric targets and DeepEval4.2.8/local4B settings are in deepeval_final_results.jsonl. Three reference-free metrics plus two reference metrics apply only when valid reference answers exist. Bounded Judge timeout/malformed results remain not_evaluated; averages count scored observations only. Safe evidence-only output is not silently scored as a successfully answered question. Human claim support and unsupported-claim rate remain null. Local provider API currency is0; total CPU/GPU/energy cost is unmeasured.\n"
    report += "\nLong-context eligibility uses the same serialized8192-unit context ceiling and all native source children. Parser-filtered children are not raw unfiltered full-paper equivalence; eligibility/exclusions are retained in long_context.csv. Fixed-evidence ordering is a small synthetic controlled test with human completeness unmeasured; it does not justify changing evidence order.\n"
    report += "\nA separate matched comparison actually generated from the full untruncated original provided paper plus a canonical citation-span catalogue versus RAG, under the same8192 context ceiling and production verifier. Two fitting papers produced four results (long_context_comparison.csv): the answerable case was supported with RAG and evidence_only with raw full paper; the no-answer-reference case was evidence_only in both modes. Human answer completeness and paired Judge scores are unmeasured; these two pairs do not establish a general long-context advantage. The earlier all-native-child run is retained separately in the cost ledger.\n"
    report += "\nCitation checks are stratified by local/Web and full-text/abstract evidence in citation_checks.csv, including actual citation counts, exact-quote hashes and canonical page bindings. Zero-citation abstentions are not successful scientific claims. DOI fields are absent in these particular generation samples; registered-DOI existence was not measured. Reference-no-answer and reference-answerable output statuses are in reference_abstention.csv; human correct/false-refusal decisions remain unknown. One reference-no-answer question produced a partial related statement, which does not establish a complete answer.\n"
    report += f"\n## Web, PDF and resources\n\nM5 routing fixtures rerun:{web['summary']}. The deliberately challenging fixture set retains wrong/missed routes; engineering/privacy gates do not certify semantic routing accuracy. Two actual connectivity probes are separate:{[(r['provider'], r['candidate_count'], [e.get('http_status') for e in r.get('events', [])]) for r in web['live']]}. Connectivity does not establish answer improvement or full-text availability.\n"
    report += f"\nAn actual production Web adapter smoke test used an isolated empty local library and run-ID-correlated traces. Observed modes:{[(r['local_only'], r['support_status'], r['external_attempts'], r['evidence_count']) for r in actual_web]}. Local-only abstained with0 external attempts. Allowed Web made4 external attempts, recovered from Semantic Scholar429 to200, received OpenAlex200, and handled a temporary-PDF timeout;16 Web records yielded5 evidence entries but no supported answer or model generation. This is actual acquisition/fault handling, not an answer-quality gain. The first unguarded Windows helper was re-executed by PDF worker spawning; its mixed traces are archived and excluded from paired results, with total duplicated usage unknown.\n"
    pdf = read("pdf_audit/summary.json")
    report += f"\nHistorical PDF packet:{pdf}. All50 original PDF hashes match; old offsets/blocks can differ after earlier parser fixes and are not automatic new errors. A new current-parser30-suspicious/20-ordinary packet is separately frozen and published with human decisions blank. Codex visually inspected double columns, simple/complex tables, captions, formulas and a page boundary; scans retain typed unsupported behavior and abstracts have null geometry. AI/source checks are not human layout/cleaning certification.\n"
    stress = read("stress.json")
    for row in stress["rows"]:
        report += f"\nSynthetic replay:{row['synthetic_points']} points×384 dimensions,{row['queries']} searches,query p50/p95={row['query_p50_ms']:.2f}/{row['query_p95_ms']:.2f}ms; Qdrant memory after={row['memory_after']['qdrant_container_memory']}, indexed vectors={row['indexed_vectors']}. These random-vector runs measure actual local search/storage observations; no embedding inference, answer quality, ANN or concurrent-user capacity claim. Collections are separate from user data.\n"
    report += "\nDocker27.3.1 was restored by preserving/recreating only its inaccessible stale runtime directory, without deleting volumes. Backed-up scoped maintenance removed181 retired vectors; current93 valid vectors and1910 foreign-scope vectors were preserved, and a second repair was idempotent. SQLite retains both papers and274 historical chunks. User data was not used for benchmark questions or screenshots.\n"
    report += "\n## Acceptance and delivery limits\n\nEngineering deliverables include reproducible tools, frozen candidates, paired ablations, original-anchor checks, AI error review, bounded Judge results, stress/connectivity diagnostics, English architecture/install/maintenance documentation and a recording/walkthrough script. The existing real UI screenshot is retained. Automatic approval rejected new frontend preview startup with only blocked-by-policy detail, so no new UI screenshot or browser walkthrough is claimed. API/developer observations are not a3–5-person user study.\n\nFormal M8 gates remain unmet:0 independent human Gold,0/50 human PDF reviews,0/10 human error reviews and unmeasured human claim support for six scenario classes. AI review delegated by the user is labelled AI. Read annotation_guidelines.md and release_gate.json; no synthetic/development result is promoted to independent human acceptance. Final review, test and exact-head CI evidence are recorded in the PR and metrics history.\n"
    report += "\nThe initial local-PDF fixture was rejected before generation because in-memory bbox tuples differed from persisted JSON arrays. A RED test reproduced the fixture-boundary error; the corrected fixture uses the actual JSON wire representation and retains its original PDF/source database. All six scenarios were rerun; prior results/costs remain separate in diagnostic_cost_ledger.json. This correction did not change the frozen production parser/registry or the40-question corpus.\n"
    report += f"\nActual model files/digests are now checked against a bound identity manifest:{model_verification}. This initial development run recorded weight identities after its first diagnostics, so prospective every-stage weight verification is not retrospectively claimed. Fresh preparation now includes model_freeze.json in the initial manifest, and retrieval/generation/Judge/report stages reject cache changes or same-name retags. Model names/config alone are insufficient.\n"
    demo_path = root / "demo_observations.json"
    if demo_path.exists():
        demo = json.loads(demo_path.read_text())
        report += f"\nActual isolated synthetic API walkthrough:{demo}. This covers backend calls/source jumps/feedback/comparison only, with0 human participants and no new browser screenshot.\n"
    (destination / "evaluation_final.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "published": str(destination),
                "human_gold": 0,
                "formal_m8_accepted": False,
                "cases": 40,
                "variants": 7,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.destination)
