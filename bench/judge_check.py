"""Check the judge against hand-labelled answers before trusting its scores.

    uv run python bench/judge_check.py [--model claude-opus-5-5] [--effort medium]
    uv run python bench/judge_check.py --provider ollama --model qwen2.5:3b   # local, free
    uv run python bench/judge_check.py --provider openai --model gemini-3.5-flash-lite \\
        --base-url https://generativelanguage.googleapis.com/v1beta/openai/   # hosted, LLM_API_KEY

Each case below is an answer written to have a known number of supported and unsupported
claims and known-good or known-bad citations. The script runs the real judge on every case
and reports how often its metrics land on the expected values.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ragspec.config import JudgeConfig, load_env
from ragspec.judge import canonicalize_citations, make_judge
from ragspec.metrics import generation_metrics
from ragspec.models import Context

ROOT = Path(__file__).parent.parent
CORPUS = {d["id"]: f"{d['title']}: {d['text']}" for d in map(
    json.loads, (ROOT / "examples" / "quickstart" / "corpus.jsonl").open())}

# (name, question, context ids, answer, expected metrics). None = metric should be undefined.
CASES = [
    ("faithful, cited", "Can I get a refund on an annual plan?",
     ["billing-refunds", "billing-plans"],
     "Yes. Annual plans can be refunded in full within 30 days of purchase [billing-refunds].",
     {"faithfulness": 1.0, "citation_precision": 1.0, "citation_recall": 1.0}),
    ("faithful, wrong citation", "Can I get a refund on an annual plan?",
     ["billing-refunds", "billing-plans"],
     "Annual plans can be refunded in full within 30 days of purchase [billing-plans].",
     {"faithfulness": 1.0, "citation_precision": 0.0, "citation_recall": 0.0}),
    ("changed number", "How long is the refund window?", ["billing-refunds"],
     "Annual plans can be refunded within 60 days of purchase [billing-refunds].",
     {"faithfulness": 0.0}),
    ("one of two claims invented", "What does Team cost?", ["billing-plans"],
     "Team costs $12 per seat per month [billing-plans]. "
     "It also includes a 14-day free trial [billing-plans].",
     {"faithfulness": 0.5, "citation_precision": 0.5, "citation_recall": 1.0}),
    ("contradiction", "Can I get 2FA codes by SMS?", ["auth-2fa"],
     "Yes, you can receive two-factor codes by SMS [auth-2fa].",
     {"faithfulness": 0.0}),
    ("no citations", "How often are backups taken?", ["data-backups"],
     "Backups run every 6 hours and are kept for 35 days.",
     {"faithfulness": 1.0, "citation_precision": None, "citation_recall": 0.0}),
    ("positional citations", "How often are backups taken?", ["data-residency", "data-backups"],
     "Data is backed up every 6 hours [2]. Backups are kept for 35 days [2].",
     {"faithfulness": 1.0, "citation_precision": 1.0, "citation_recall": 1.0}),
    ("abstains", "Is there a Jira integration?", ["integrations-slack", "integrations-github"],
     "The help center doesn't mention a Jira integration, so I can't confirm one exists. "
     "Please contact support.",
     {"faithfulness": None}),
    ("outside knowledge", "Is SAML supported?", ["auth-sso"],
     "SSO supports SAML 2.0 [auth-sso]. SAML was first published by OASIS in 2002 [auth-sso].",
     {"faithfulness": 0.5, "citation_precision": 0.5}),
    ("derived arithmetic", "What credit do we get at 99.75% uptime?", ["status-uptime"],
     "The SLA target is 99.95% [status-uptime]. At 99.75% you are 0.2% below target, which is "
     "two 0.1% steps, so the credit is 20% of the monthly fee [status-uptime].",
     {"faithfulness": 1.0, "citation_precision": 1.0}),
    # Added after real runs showed judges rejecting correct facts and approving wrong arithmetic.
    ("correct: names the product, source doesn't", "Do guests use a seat?", ["projects-guests"],
     "Guests do not use a paid seat in Nimbus.",
     {"faithfulness": 1.0}),
    ("correct: product name in passing", "What is the upload limit?", ["projects-limits"],
     "You can upload individual files up to 5 GB in size to your Nimbus projects.",
     {"faithfulness": 1.0}),
    ("correct: product name in passing", "Which plan do I need for SAML?", ["auth-sso"],
     "You'll need the Enterprise plan to use SAML single sign-on with Nimbus.",
     {"faithfulness": 1.0}),
    ("correct: first-person voice", "How fast is urgent support?", ["support-hours"],
     "We give Enterprise customers a first response within 1 hour for urgent issues.",
     {"faithfulness": 1.0}),
    ("correct: calculated number", "What credit do we get at 99.75% uptime?", ["status-uptime"],
     "If uptime drops to 99.75%, you receive service credits equal to 20% of the monthly fee.",
     {"faithfulness": 1.0}),
    ("wrong: calculated number", "What credit do we get at 99.75% uptime?", ["status-uptime"],
     "If uptime drops to 99.75%, you receive service credits equal to 1.5% of the monthly fee.",
     {"faithfulness": 0.0}),
    ("wrong: invented feature naming the product", "Is there a mobile app?",
     ["integrations-slack"],
     "Nimbus offers a mobile app for iOS.",
     {"faithfulness": 0.0}),
]

# Kept out of tuning: run these only to measure a finished change, and never adjust the prompt
# or finalize() in response to an individual result here. Each answer's expected score was set
# by hand from the source document.
HOLDOUT = [
    ("correct: product name", "How long are audit logs kept?", ["data-retention"],
     "Nimbus keeps audit logs for 7 years on Enterprise.", {"faithfulness": 1.0}),
    ("correct: product name", "What is the default page size?", ["api-pagination"],
     "In Nimbus, page size defaults to 50.", {"faithfulness": 1.0}),
    ("correct: product name", "How often is data backed up?", ["data-backups"],
     "Your Nimbus workspace data is backed up every 6 hours.", {"faithfulness": 1.0}),
    ("correct: first-person voice", "Are backups encrypted?", ["data-backups"],
     "We encrypt backups with AES-256.", {"faithfulness": 1.0}),
    ("correct: product name, cited", "How long are backups kept?", ["data-backups"],
     "Nimbus retains backups for 35 days [data-backups].",
     {"faithfulness": 1.0, "citation_precision": 1.0, "citation_recall": 1.0}),
    ("correct: calculated number", "How many backups run per day?", ["data-backups"],
     "All workspace data is backed up 4 times a day.", {"faithfulness": 1.0}),
    ("correct: calculated number", "What credit do we get at 99.65% uptime?", ["status-uptime"],
     "If uptime is 99.65%, the service credit is 30% of the monthly fee.", {"faithfulness": 1.0}),
    ("correct: calculated number", "What credit do we get at 99.05% uptime?", ["status-uptime"],
     "If uptime is 99.05%, the service credit is 50% of the monthly fee.", {"faithfulness": 1.0}),
    ("wrong: calculated number", "What credit do we get at 99.65% uptime?", ["status-uptime"],
     "If uptime is 99.65%, the service credit is 3% of the monthly fee.", {"faithfulness": 0.0}),
    ("wrong: calculated number", "What credit do we get at 99.05% uptime?", ["status-uptime"],
     "If uptime is 99.05%, the service credit is 90% of the monthly fee.", {"faithfulness": 0.0}),
    ("wrong: calculated number", "How many backups run per day?", ["data-backups"],
     "All workspace data is backed up 6 times a day.", {"faithfulness": 0.0}),
    ("wrong: invented, names product", "Is there an Android app?", ["auth-sessions"],
     "Nimbus offers a native Android app.", {"faithfulness": 0.0}),
    ("wrong: invented, names product", "Does it work with Teams?", ["integrations-slack"],
     "Nimbus integrates with Microsoft Teams.", {"faithfulness": 0.0}),
    ("wrong: invented, names product", "When was the company founded?", ["billing-plans"],
     "Nimbus was founded in 2018.", {"faithfulness": 0.0}),
    ("wrong: changed fact, names product", "What is the Team rate limit?", ["api-rate-limits"],
     "The Nimbus API allows 500 requests per minute on Team.", {"faithfulness": 0.0}),
    ("wrong: changed fact, names product", "Which plan has EU hosting?", ["data-residency"],
     "Nimbus offers EU hosting on the Team plan.", {"faithfulness": 0.0}),
    ("wrong: changed fact, names product", "Can I get 2FA codes by SMS?", ["auth-2fa"],
     "Nimbus sends two-factor codes by SMS.", {"faithfulness": 0.0}),
    ("wrong: citation", "How long are backups kept?", ["data-backups", "data-retention"],
     "Nimbus retains backups for 35 days [data-retention].",
     {"faithfulness": 1.0, "citation_precision": 0.0}),
    ("mixed: one of two claims wrong", "What can guests do?", ["projects-guests"],
     "Guests can edit tasks in projects they are invited to. Guests can also create projects.",
     {"faithfulness": 0.5}),
    ("abstains, names product", "Is there a Zapier integration?", ["integrations-slack"],
     "The help center doesn't say whether Nimbus has a Zapier integration, so please contact "
     "support.", {"faithfulness": None}),
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "ollama", "openai"])
    parser.add_argument("--base-url")
    parser.add_argument("--product", default="Nimbus")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--set", dest="case_set", default="dev", choices=["dev", "holdout"])
    parser.add_argument("--summary-only", action="store_true",
                        help="Only totals per group, so a baseline can't influence tuning")
    args = parser.parse_args()
    cases = HOLDOUT if args.case_set == "holdout" else CASES
    load_env()
    judge = make_judge(JudgeConfig(provider=args.provider, model=args.model, effort=args.effort,
                                   base_url=args.base_url, product=args.product))
    # Local models serve one request at a time, and free hosted tiers rate-limit hard.
    sem = asyncio.Semaphore(10 if args.provider == "anthropic" else 1)

    async def one(case):
        name, question, ids, answer, expected = case
        contexts = [Context(id=i, text=CORPUS[i]) for i in ids]
        canonical, cited = canonicalize_citations(answer, contexts, r"\[([^\[\]]+)\]", "auto")
        async with sem:
            result = await judge.judge(question, contexts, canonical, cited)
        return name, expected, generation_metrics(result), result

    results = await asyncio.gather(*(one(c) for c in cases))
    checks = passed = errors = 0
    groups: dict[str, list[int]] = {}
    for name, expected, got, result in results:
        misses = []
        for metric, want in expected.items():
            have = got[metric]
            ok = (have is None) if want is None else (have is not None and abs(have - want) < 0.01)
            checks += 1
            passed += ok
            if not ok:
                misses.append(f"{metric}: expected {want}, got {have}")
        errors += bool(result.error)
        group = groups.setdefault(name.split(":")[0] if ":" in name else "other", [0, 0])
        group[0] += not misses
        group[1] += 1
        if args.summary_only:
            continue
        status = "PASS" if not misses else "FAIL"
        print(f"{status}  {name}" + (f"  ({result.error})" if result.error else ""))
        for m in misses:
            print(f"        {m}")
            for c in result.claims:
                print(f"          - [{c.verdict}] {c.text} cited={c.cited_ids} "
                      f"supporting={c.supporting_ids}")
    print(f"\n{passed}/{checks} metric checks match across {len(cases)} cases ({errors} errors)")
    for group, (ok, total) in sorted(groups.items()):
        print(f"  {group}: {ok}/{total} cases")


if __name__ == "__main__":
    asyncio.run(main())
