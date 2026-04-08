# D1 Hypothesis Report

- Dataset: `/Users/kazdoraw/developer/med-agent/study/data/d1_messages_v5_test.csv`
- Samples: `1800`
- L1 target macro-F1: `0.85`
- L2 target macro-F1: `0.90`

## Verdict

- L1 macro-F1: `0.9732` -> PASS
- L2 macro-F1: `0.9632` -> PASS
- Total errors (L1 or L2): `82`

## Metrics

| Level | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| L1 | 0.9739 | 0.9732 | 0.9740 |
| L2 | 0.9628 | 0.9632 | 0.9632 |

## Top L2 Error Pairs

| Gold | Pred | Count |
|---|---|---:|
| unclear | unclear | 4 |
| reschedule | new_appointment | 3 |
| clinic_info | unclear | 3 |
| followup | unclear | 3 |
| complaint | unclear | 2 |
| unclear | gratitude | 2 |
| visit_prep | followup | 2 |
| procedure | negative_general | 2 |
| followup | followup | 2 |
| symptom | complaint | 2 |

## Dataset Distribution

- L1: `{'conversational': 375, 'feedback': 600, 'booking': 225, 'anamnesis': 225, 'faq': 375}`
- L2 classes: `24`

## Artifacts

- [l1_metrics.json](/Users/kazdoraw/developer/med-agent/study/outputs/hypothesis_d1/tables/l1_metrics.json)
- [l2_metrics.json](/Users/kazdoraw/developer/med-agent/study/outputs/hypothesis_d1/tables/l2_metrics.json)
- [d1_error_analysis.csv](/Users/kazdoraw/developer/med-agent/study/outputs/hypothesis_d1/tables/d1_error_analysis.csv)
- [d1_confusion_l1.png](/Users/kazdoraw/developer/med-agent/study/outputs/hypothesis_d1/figures/d1_confusion_l1.png)
- [d1_confusion_l2.png](/Users/kazdoraw/developer/med-agent/study/outputs/hypothesis_d1/figures/d1_confusion_l2.png)
- [d1_f1_l1.png](/Users/kazdoraw/developer/med-agent/study/outputs/hypothesis_d1/figures/d1_f1_l1.png)
- [d1_f1_l2.png](/Users/kazdoraw/developer/med-agent/study/outputs/hypothesis_d1/figures/d1_f1_l2.png)
- [d1_reliability_l2.png](/Users/kazdoraw/developer/med-agent/study/outputs/hypothesis_d1/figures/d1_reliability_l2.png)
