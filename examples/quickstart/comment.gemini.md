<!-- ragspec-report -->
## ragspec: `PR #1` vs `main`

❌ **2 gate failures**

- faithfulness dropped -0.083 (95% CI -0.154 to -0.025, n=20)
- citation_recall dropped -1.000 (95% CI -1.000 to -1.000, n=20)

| | Metric | Base | Head | Δ | 95% CI | ↑ / ↓ |
|---|---|---:|---:|---:|---|---:|
| ⚪ | recall@1 | 0.688 | 0.688 | +0.000 | +0.000 … +0.000 | 0 / 0 |
| ⚪ | recall@4 | 0.875 | 0.812 | -0.062 | -0.167 … +0.000 | 0 / 2 |
| ⚪ | ndcg@1 | 0.708 | 0.708 | +0.000 | +0.000 … +0.000 | 0 / 0 |
| ⚪ | ndcg@4 | 0.800 | 0.771 | -0.029 | -0.073 … +0.000 | 0 / 2 |
| ⚪ | mrr | 0.792 | 0.771 | -0.021 | -0.052 … +0.000 | 0 / 2 |
| 🔴 | faithfulness | 1.000 | 0.905 | -0.083 | -0.154 … -0.025 | 0 / 5 |
| 🔴 | citation_recall | 1.000 | 0.000 | -1.000 | -1.000 … -1.000 | 0 / 20 |
| ⚪ | abstention_rate _(info)_ | 0.154 | 0.192 | – | – | 0 / 0 |
| ➖ | citation_precision | 1.000 | – | – | – | 0 / 0 |
| ⏱ | latency p50 / p95 (ms) | 1306 / 38476 | 1493 / 68628 | | | |

<details><summary>Biggest question-level drops (10)</summary>

| Question | Metric | Base | Head |
|---|---|---:|---:|
| `audit-retention` How long are audit logs kept on Enterprise? | citation_recall | 1.00 | 0.00 |
| `backup-frequency` How often is data backed up and for how long are backups kept? | citation_recall | 1.00 | 0.00 |
| `export-link` How long is the workspace export download link valid? | citation_recall | 1.00 | 0.00 |
| `github-autoclose` Can tasks close automatically when a PR is merged? | citation_recall | 1.00 | 0.00 |
| `guest-seat` Do guests count as paid seats? | citation_recall | 1.00 | 0.00 |
| `india-hosting` Can our data be stored in India? | citation_recall | 1.00 | 0.00 |
| `page-size` What's the maximum page size for list endpoints? | citation_recall | 1.00 | 0.00 |
| `rate-limit` What happens if my integration sends more than 1,000 API requests a minute on the Team … | citation_recall | 1.00 | 0.00 |
| `refund-annual` Can I get my money back on an annual plan? | citation_recall | 1.00 | 0.00 |
| `refund-monthly` I'm on a monthly plan. If I cancel, do I get a refund? | citation_recall | 1.00 | 0.00 |

</details>

<sub>26 questions · 🔴 significant drop · 🟢 significant gain · ↑/↓ = questions that improved/regressed</sub>
