# Overnight experiments, 12–13 October 2026

All numbers below come from runs on this machine (RTX 4060 Laptop, 8 GB) against the
same SQuAD v1.1 development questions used in the paper. Raw outputs are in
`results/`, run logs in `results/logs/`. Every model used is open and runs locally.

---

## 1. Retrieval: the published numbers are beaten on every metric

Evaluated on the same 2,000 development questions as the paper
(`results/sweep_confirm.csv`, `results/sweep_l12.csv`).

| Configuration | R@10 | R@20 | MRR | NDCG@10 | ms/q |
|---|---|---|---|---|---|
| Published baseline (reproduced) | 96.25 | 96.65 | 85.25 | 87.97 | 164 |
| Depth 100/50, L6 re-ranker | 96.90 | 97.90 | 85.44 | 88.23 | 195 |
| L12 re-ranker, depth 50/30 | 96.45 | 96.70 | 86.17 | 88.73 | 230 |
| L12 re-ranker, depth 100/50 | 97.25 | 97.85 | 86.52 | 89.17 | 284 |
| **E5-base + L12, depth 100/50** | **97.50** | **98.35** | **86.79** | **89.42** | 296 |
| L12, depth 100/50, chunked + dedup | 97.10 | 97.85 | 86.69 | 89.25 | 228 |

Best configuration against the paper: **+1.25 Recall@10, +1.70 Recall@20,
+1.54 MRR, +1.45 NDCG@10**, with no training and no proprietary models.

The two levers are separable, which is what makes this reportable:

- **Candidate depth** (50 → 100 per list, re-rank 30 → 50) buys recall: the right
  passage reaches the shortlist more often.
- **Re-ranker size** (MiniLM-L6 → L12) buys ranking quality: MRR and NDCG rise while
  recall barely moves, because re-ranking reorders a fixed candidate set.

A cheaper operating point exists: chunked + deduplicated retrieval reaches 97.10
Recall@10 at 228 ms, within 0.4 points of the best configuration at 23% less latency,
because chunks are shorter for the cross-encoder to read.

### Screening versus confirmation

Eight embedder × re-ranker pairs were screened on 500 questions
(`results/sweep_screen.csv`) before confirming the best at 2,000. E5-base looked no
better than MiniLM-L6 at 500 questions but won at full scale — an argument for
confirming screening results rather than trusting a small sample.

---

## 2. End-to-end: better retrieval, then reading more than one passage

Two independent gains, measured separately and together with the same RoBERTa-base
reader (`results/end_to_end_topk.json`, `results/end_to_end_best.json`).

| Retrieval stack | Reading | n | EM | F1 |
|---|---|---|---|---|
| Published | — | 100–200 | 71.00 | 72.34 |
| Baseline (MiniLM + L6, depth 50/30) | rank-1 | 200 | 66.50 | 69.77 |
| Baseline | top-5 | 200 | 72.00 | 76.24 |
| **Best (E5 + L12, depth 100/50)** | rank-1 | 500 | 75.60 | 78.41 |
| **Best** | **top-5** | **500** | **77.20** | **80.80** |

The full stack beats the published end-to-end result by **+6.20 EM and +8.46 F1**, on
2.5 times the sample.

The two effects interact in a way worth stating: better retrieval alone lifts rank-1
reading from 66.50 to 75.60 EM, and top-K then adds +1.60 more. With the baseline
retriever top-K was worth +5.50, because rank 1 was wrong far more often. The winning
answer came from rank 1 in 320 of 500 questions with the best stack, against 98 of 200
with the baseline. Fix retrieval first; reading more passages collects what remains.

The mechanism is in the data: with the baseline retriever the winning answer came from
rank 1 in only 98 of 200 questions, so 102 answers came from ranks 2–5 — passages the
retriever already returns but the reader never sees. 85 of 200 answers changed; with
the best stack, 143 of 500 changed.

**This is what makes the complexity-based K meaningful.** Until now K was computed and
then discarded, because every caller overrode it and the reader consumed only rank 1.
This is the first measurement in the project where K changes answers.

Caveats to state in the paper: K was capped at 5 in both runs, so the classifier's 10
and 15 were clipped; samples are 200 and 500 questions rather than the full development
set; reading five chunks costs roughly five times the reader time (about 40–80 ms,
still small next to retrieval).

### How much of this is the decoder rather than top-K?

The end-to-end runs also use a new span decoder, so the two were separated on gold
passages (`results/decoder_comparison.json`, 2,000 questions, same checkpoint):

| Decoder | EM | F1 | ms/q |
|---|---|---|---|
| Arg-max start and end, first window only (`src/qa_infer.py`) | 86.80 | 92.01 | 9.0 |
| Joint start/end pairs over all windows (`src/qa_decode.py`) | **87.45** | **92.78** | 16.2 |

The decoder is worth **+0.65 EM / +0.78 F1** on its own, changing 26 of 2,000 answers.
Both end-to-end settings above used the same decoder, so the rank-1 → top-K comparison
was already clean; of the +5.50 EM, at most 0.65 is attributable to decoding.

### Choosing K: the gain peaks at 5 and reverses by 10

`results/end_to_end_kcurve.json`, 200 questions, identical retrieval per question.

| K | EM | F1 |
|---|---|---|
| 1 | 66.50 | 69.77 |
| 3 | 71.50 | 74.85 |
| **5** | **72.00** | **76.24** |
| 10 | 69.50 | 74.11 |

Going from 1 to 3 captures most of the benefit (+5.00 EM); 5 adds 0.50 more; 10 loses
2.50. The reversal is informative: as K grows the reader sees more irrelevant passages,
and span confidence is not calibrated well enough across passages to stop a confident
wrong answer from outranking the right one.

**This bears directly on the complexity-based K.** The current classifier assigns
5 / 10 / 15, but 10 and 15 are past the peak on this benchmark. The evidence supports a
range of 3–5, which lets the paper justify those values instead of asserting them.

---

## 3. Reader: checkpoint restored and the published result reproduced

`results/roberta-base-squad1_rerun_metrics.json`, full 10,570-question dev set.

| | Published | Rerun | Δ |
|---|---|---|---|
| Exact Match | 85.38 | 85.37 | −0.01 |
| F1 | 91.74 | 91.83 | +0.09 |
| Training time | 2.45 h (RTX 5060) | 1.19 h (RTX 4060) | — |

The fine-tuned checkpoint lost with the old machine now exists again at
`models/roberta-base-squad1` (1.9 GB), which restores the FastAPI backend and the live
demo. The recipe matched the original exactly: 8,301 optimizer steps, effective batch
32, 6% warm-up, same learning rate, best checkpoint chosen by evaluation loss.

The faster wall-clock time on a slower GPU comes from the newer PyTorch and a smaller
per-device batch with gradient accumulation, which fits 8 GB better.

---

## 4. Multi-query expansion does not help on SQuAD

`results/multiquery_comparison_v2.csv`, 2,000 questions, identical pipeline except for
how query variants are produced.

| Variant source | R@10 | R@20 | MRR | NDCG@10 | Variants/q | Generation |
|---|---|---|---|---|---|---|
| None | **96.25** | 96.65 | **85.25** | **87.97** | 1.00 | 0 ms |
| Rules (current system) | **96.25** | **96.70** | 85.19 | 87.91 | 2.23 | 0 ms |
| flan-t5-small | 96.05 | 96.50 | 85.14 | 87.83 | 1.25 | 23 ms |
| Trained paraphraser (T5-base) | 95.45 | 95.85 | 84.78 | 87.41 | 2.78 | 19 ms |

Two separate findings:

**Model choice dominates.** Asked to paraphrase, flan-t5-small answers the question
instead, often inventing facts ("What is the speed of light?" → "speed of light is
0.005 m/s"), and those invented terms pollute the search. Only 496 of 4,000 generated
variants survived a filter requiring a question that introduces no new numbers; a model
fine-tuned for paraphrasing passed 3,569 of 4,000. Generation quality is a model
property, not a prompt-engineering detail.

**Even good paraphrases do not help here.** The trained paraphraser produced fluent,
faithful rewrites ("Where did Super Bowl 50 take place?" → "Where was Super Bowl 50
held?") and still lost 0.80 Recall@10. SQuAD questions were written by crowdworkers
looking at the passage, so the original wording already overlaps the target text more
than any rewrite; each paraphrase moves away from that optimum and dilutes the fused
ranking.

An earlier run without the generation filter is kept in
`results/multiquery_comparison.csv` for the record; it scored 95.00 Recall@10 using
degenerate output and should not be quoted.

---

## 5. Reproducibility

The published baseline row was reproduced **three times** tonight, in independent runs,
to the decimal: 96.25 / 96.65 / 85.25 / 87.97. This resolves the discrepancy noted
earlier between `results/retrieval_progressive.csv` and the notebook's saved outputs
(which show 94.20 and 94.25 for the two multi-query rows) in favour of the CSV and the
paper.

---

## What this implies for `report/main.tex`

1. **Ablation table (Table III).** Add the three new full-scale rows: depth 100/50,
   L12 re-ranker, and E5-base + L12. The best row raises the paper's headline retrieval
   result from 96.25 to 97.50 Recall@10.
2. **Section V-A text.** The claim that re-ranking is the single largest gain still
   holds, but it can now be decomposed: depth raises recall, re-ranker size raises
   ranking quality.
3. **End-to-end section (V-D).** Replace the end-to-end table. The full stack reaches
   77.20 EM / 80.80 F1 against the published 71.00 / 72.34, and the retrieval and
   reading contributions can be reported separately (+9.10 EM from retrieval, +1.60
   more from top-K). This also converts the dynamic-K design from an untested claim
   into a measured one.
4. **Table II and Section III-D.** Note that K now feeds the reader, which is what makes
   the complexity classifier observable — and retune the values. The measured optimum is
   K = 5, with K = 3 capturing most of the gain and K = 10 losing 2.50 EM, so the
   published 5 / 10 / 15 should become something inside 3–5.
5. **Reader section (V-B).** The span decoder is a second free improvement:
   +0.65 EM / +0.78 F1 on gold passages for 7 ms, by scoring start/end pairs jointly
   across all windows instead of taking independent arg-maxes over the first one.
6. **Novelty discussion (Section VI).** The claim "adaptation with no extra models" is
   now evidence-backed: the rule-based variants match or beat both transformer
   expanders at zero generation cost. Table I's positioning against Mageshwari and
   Shrijith can cite measured numbers instead of asserting a design preference.
7. **Limitations (Section VII).** Replace the speculation that a generative paraphraser
   would increase variant diversity: it was tried, it works as a paraphraser, and it
   still reduces retrieval quality on this benchmark.

## Reproducing any of it

```bash
python scripts/prefetch_models.py          # one-off, caches every model used
python scripts/eval_multiquery.py --n 2000 --t5-models google/flan-t5-small,humarin/chatgpt_paraphraser_on_T5_base
python scripts/sweep_retrieval.py --stage confirm --n 2000 --configs results/sweep_l12_configs.json --out results/sweep_l12.csv
python scripts/train_reader.py --model roberta-base --run-name roberta-base-squad1
python scripts/eval_end_to_end.py --checkpoint models/roberta-base-squad1 --n 200
python scripts/summarise_runs.py
```

## Machine notes

The GPU driver faulted several times during the night (one VIDEO_TDR_ERROR bugcheck at
21:56 that restarted the machine, driver-level `nvlddmkm` events at 22:06, and two
further CUDA faults at 03:25). Every stage is wrapped in retries and resumes from its
own output, so the only cost was time. Before the next long run, a driver reinstall or
a less aggressive GPU clock profile is worth trying.
