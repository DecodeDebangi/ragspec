<!-- ragspec-report -->
## ragspec: `scifact-rerank-minilm` vs `scifact-hybrid`

❌ **3 gate failures**

- recall@10 dropped -0.048 (95% CI -0.078 to -0.018, n=300)
- ndcg@10 dropped -0.039 (95% CI -0.066 to -0.012, n=300)
- mrr dropped -0.035 (95% CI -0.067 to -0.002, n=300)

| | Metric | Base | Head | Δ | 95% CI | ↑ / ↓ |
|---|---|---:|---:|---:|---|---:|
| ⚪ | recall@1 | 0.572 | 0.544 | -0.028 | -0.072 … +0.016 | 23 / 32 |
| ⚪ | recall@5 | 0.770 | 0.753 | -0.017 | -0.054 … +0.019 | 20 / 22 |
| 🔴 | recall@10 | 0.855 | 0.808 | -0.048 | -0.078 … -0.018 | 8 / 22 |
| ⚪ | ndcg@1 | 0.597 | 0.567 | -0.030 | -0.077 … +0.017 | 23 / 32 |
| ⚪ | ndcg@5 | 0.692 | 0.666 | -0.026 | -0.059 … +0.005 | 45 / 56 |
| 🔴 | ndcg@10 | 0.723 | 0.685 | -0.039 | -0.066 … -0.012 | 46 / 67 |
| 🔴 | mrr | 0.688 | 0.653 | -0.035 | -0.067 … -0.002 | 43 / 63 |
| ⏱ | latency p50 / p95 (ms) | 31 / 36 | 181 / 202 | | | |

<details><summary>Biggest question-level drops (10)</summary>

| Question | Metric | Base | Head |
|---|---|---:|---:|
| `1245` The one-child policy has been successful in lowering population growth. | recall@10 | 1.00 | 0.00 |
| `1280` The ureABIEFGH gene cluster encodes urease maturation proteins : UreD/UreH, UreE, UreF,… | recall@10 | 1.00 | 0.00 |
| `1368` Vitamin D deficiency effects the term of delivery. | recall@10 | 1.00 | 0.00 |
| `239` Cellular aging closely links to an older appearance. | recall@10 | 1.00 | 0.00 |
| `312` De novo assembly of sequence data has more specific contigs than unassembled sequence d… | recall@10 | 1.00 | 0.00 |
| `314` Deamination of cytidine to uridine on the minus strand of viral DNA results in catastro… | recall@10 | 1.00 | 0.00 |
| `384` Epidemiological disease burden from noncommunicable diseases is more prevalent in low e… | recall@10 | 1.00 | 0.00 |
| `415` Female carriers of the Apolipoprotein E4 (APOE4) allele have increased risk for dementia. | recall@10 | 1.00 | 0.00 |
| `517` High levels of copeptin decrease risk of diabetes. | recall@10 | 1.00 | 0.00 |
| `577` In mice, P. chabaudi parasites are able to proliferate faster early in infection when i… | recall@10 | 1.00 | 0.00 |

</details>

<details><summary>Biggest improvements (5)</summary>

| Question | Metric | Base | Head |
|---|---|---:|---:|
| `1049` Ribosomopathies have a low degree of cell and tissue specific pathology. | recall@10 | 0.00 | 1.00 |
| `1200` The binding orientation of the ML-SA1 activator at hTRPML2 is different from the bindin… | recall@10 | 0.00 | 1.00 |
| `1221` The genomic aberrations found in matasteses are very similar to those found in the prim… | recall@10 | 0.00 | 1.00 |
| `775` Mice defective for deoxyribonucleic acid (DNA) polymerase I (polI) reveal increased sen… | recall@10 | 0.00 | 1.00 |
| `1088` Silencing of Bcl2 is important for the maintenance and progression of tumors. | recall@5 | 0.00 | 1.00 |

</details>

<sub>300 questions · 🔴 significant drop · 🟢 significant gain · ↑/↓ = questions that improved/regressed</sub>
