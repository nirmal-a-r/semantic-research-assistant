# Overnight experiment summary

## Full-scale runs (2,000 questions — comparable with the paper)

| config                                                              |   n_questions |   Recall@10 |   Recall@20 |   MRR |   NDCG@10 |   ms_per_question |
|:--------------------------------------------------------------------|--------------:|------------:|------------:|------:|----------:|------------------:|
| e5-base-v2|ms-marco-MiniLM-L12-v2|c100|r50|passage|single           |          2000 |       97.50 |       98.35 | 86.79 |     89.42 |            296.42 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L12-v2|c100|r50|passage|single     |          2000 |       97.25 |       97.85 | 86.52 |     89.17 |            283.74 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L12-v2|c100|r50|chunk|single|dedup |          2000 |       97.10 |       97.85 | 86.69 |     89.25 |            228.25 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L6-v2|c100|r50|passage|single      |          2000 |       96.90 |       97.90 | 85.44 |     88.23 |            195.12 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L12-v2|c50|r30|passage|single      |          2000 |       96.45 |       96.70 | 86.17 |     88.73 |            229.89 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L6-v2|c50|r30|passage|mq           |          2000 |       96.25 |       96.70 | 85.19 |     87.91 |            293.65 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L6-v2|c50|r30|passage|single       |          2000 |       96.25 |       96.65 | 85.25 |     87.97 |            162.68 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L6-v2|c50|r30|chunk|single|dedup   |          2000 |       96.10 |       96.45 | 85.26 |     87.95 |            160.32 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L6-v2|c50|r30|chunk|single         |          2000 |       96.00 |       96.45 | 85.20 |     87.32 |            164.99 |

- Recall@10: 97.50 (e5-base-v2|ms-marco-MiniLM-L12-v2|c100|r50|passage|single) — beats published best 96.25 by 1.25
- Recall@20: 98.35 (e5-base-v2|ms-marco-MiniLM-L12-v2|c100|r50|passage|single) — beats published best 96.70 by 1.65
- MRR: 86.79 (e5-base-v2|ms-marco-MiniLM-L12-v2|c100|r50|passage|single) — beats published best 85.25 by 1.54
- NDCG@10: 89.42 (e5-base-v2|ms-marco-MiniLM-L12-v2|c100|r50|passage|single) — beats published best 87.97 by 1.45

## Screening runs (500 questions — not comparable with the paper)

| config                                                          |   n_questions |   Recall@10 |   Recall@20 |   MRR |   NDCG@10 |   ms_per_question |
|:----------------------------------------------------------------|--------------:|------------:|------------:|------:|----------:|------------------:|
| e5-base-v2|ms-marco-MiniLM-L12-v2|c50|r30|passage|single        |           500 |       97.60 |       97.80 | 84.09 |     87.46 |            275.78 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L12-v2|c50|r30|passage|single  |           500 |       97.60 |       97.80 | 83.96 |     87.37 |            264.40 |
| bge-small-en-v1.5|ms-marco-MiniLM-L12-v2|c50|r30|passage|single |           500 |       96.80 |       97.00 | 83.81 |     87.05 |            275.52 |
| bge-base-en-v1.5|ms-marco-MiniLM-L12-v2|c50|r30|passage|single  |           500 |       96.60 |       97.00 | 83.51 |     86.78 |            273.42 |
| e5-base-v2|ms-marco-MiniLM-L6-v2|c50|r30|passage|single         |           500 |       97.20 |       97.80 | 82.49 |     86.14 |            202.30 |
| all-MiniLM-L6-v2|ms-marco-MiniLM-L6-v2|c50|r30|passage|single   |           500 |       97.60 |       97.80 | 82.40 |     86.19 |            175.98 |
| bge-small-en-v1.5|ms-marco-MiniLM-L6-v2|c50|r30|passage|single  |           500 |       96.80 |       97.00 | 82.28 |     85.89 |            201.27 |
| bge-base-en-v1.5|ms-marco-MiniLM-L6-v2|c50|r30|passage|single   |           500 |       96.20 |       97.00 | 82.05 |     85.56 |            199.97 |

## Multi-query: rules vs transformers (final, filtered generation)

|                                                |   Recall@10 |   Recall@20 |   MRR |   NDCG@10 |   mean_variants |   variant_gen_ms_per_question |   ms_per_question |
|:-----------------------------------------------|------------:|------------:|------:|----------:|----------------:|------------------------------:|------------------:|
| + Re-ranking (no multi-query)                  |       96.25 |       96.65 | 85.25 |     87.97 |            1.00 |                          0.00 |            156.78 |
| + Multi-query (rules)                          |       96.25 |       96.70 | 85.19 |     87.91 |            2.23 |                          0.00 |            275.14 |
| + Multi-query (flan-t5-small)                  |       96.05 |       96.50 | 85.14 |     87.83 |            1.25 |                         23.07 |            200.00 |
| + Multi-query (chatgpt_paraphraser_on_T5_base) |       95.45 |       95.85 | 84.78 |     87.41 |            2.78 |                         18.96 |            353.20 |

## Multi-query: first run (unfiltered generation, kept for the record)

|                               |   Recall@10 |   Recall@20 |   MRR |   NDCG@10 |   mean_variants |   variant_gen_ms_per_question |   ms_per_question |
|:------------------------------|------------:|------------:|------:|----------:|----------------:|------------------------------:|------------------:|
| + Re-ranking (no multi-query) |       96.25 |       96.65 | 85.25 |     87.97 |            1.00 |                          0.00 |            177.94 |
| + Multi-query (rules)         |       96.25 |       96.70 | 85.19 |     87.91 |            2.23 |                          0.00 |            336.90 |
| + Multi-query (FLAN-T5)       |       95.00 |       95.60 | 84.53 |     87.10 |            2.26 |                         52.16 |            324.80 |

## Reader rerun: roberta-base-squad1

- EM 85.37 (-0.01 vs published 85.38)
- F1 91.83 (+0.09 vs published 91.74)
- 1.19 h on NVIDIA GeForce RTX 4060 Laptop GPU

## End-to-end reading

- rank-1 only: EM 66.50  F1 69.77
- top-K (mean K=5.0): EM 72.00  F1 76.24
- published: EM 71.00  F1 72.34
- answers changed by top-K reading: 85/200
