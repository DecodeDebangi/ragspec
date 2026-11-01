<!-- ragspec-report -->
## ragspec: `PR #1` vs `main`

❌ **3 gate failures**

- faithfulness dropped -0.127 (95% CI -0.262 to -0.024, n=21)
- faithfulness = 0.797 is below the minimum 0.8
- citation_recall dropped -0.975 (95% CI -1.000 to -0.925, n=20)

| | Metric | Base | Head | Δ | 95% CI | ↑ / ↓ |
|---|---|---:|---:|---:|---|---:|
| ⚪ | recall@1 | 0.688 | 0.688 | +0.000 | +0.000 … +0.000 | 0 / 0 |
| ⚪ | recall@4 | 0.875 | 0.812 | -0.062 | -0.167 … +0.000 | 0 / 2 |
| ⚪ | ndcg@1 | 0.708 | 0.708 | +0.000 | +0.000 … +0.000 | 0 / 0 |
| ⚪ | ndcg@4 | 0.800 | 0.771 | -0.029 | -0.073 … +0.000 | 0 / 2 |
| ⚪ | mrr | 0.792 | 0.771 | -0.021 | -0.052 … +0.000 | 0 / 2 |
| 🔴 | faithfulness ⛔ | 1.000 | 0.797 | -0.127 | -0.262 … -0.024 | 0 / 4 |
| 🔴 | citation_recall | 0.977 | 0.000 | -0.975 | -1.000 … -0.925 | 0 / 20 |
| ⚪ | abstention_rate _(info)_ | 0.154 | 0.115 | – | – | 0 / 0 |
| ➖ | citation_precision | 1.000 | – | – | – | 0 / 0 |
| ⏱ | latency p50 / p95 (ms) | 28157 / 43821 | 27074 / 50895 | | | |

<details><summary>Biggest question-level drops (10)</summary>

| Question | Metric | Base | Head |
|---|---|---:|---:|
| `team-price` How much is the Team plan? | faithfulness | 1.00 | 0.00 |
| `audit-retention` How long are audit logs kept on Enterprise? | citation_recall | 1.00 | 0.00 |
| `backup-frequency` How often is data backed up and for how long are backups kept? | citation_recall | 1.00 | 0.00 |
| `export-link` How long is the workspace export download link valid? | citation_recall | 1.00 | 0.00 |
| `github-autoclose` Can tasks close automatically when a PR is merged? | citation_recall | 1.00 | 0.00 |
| `guest-seat` Do guests count as paid seats? | citation_recall | 1.00 | 0.00 |
| `india-hosting` Can our data be stored in India? | citation_recall | 1.00 | 0.00 |
| `page-size` What's the maximum page size for list endpoints? | citation_recall | 1.00 | 0.00 |
| `rate-limit` What happens if my integration sends more than 1,000 API requests a minute on the Team … | citation_recall | 1.00 | 0.00 |
| `refund-annual` Can I get my money back on an annual plan? | citation_recall | 1.00 | 0.00 |

</details>

<sub>26 questions · 🔴 significant drop · 🟢 significant gain · ↑/↓ = questions that improved/regressed</sub>
