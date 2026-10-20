# pwdet corpus v1: pwdet 0.2.0

108 items over 4 families (olmo2-1b, qwen3-0.6b, qwen3-1.7b, smollm2-360m): 61 tampered, 47 benign or presumed benign. Corpus digest `152c0ab9ffa1466d`.

A model counts as **flagged** when its verdict is suspicious, likely_tampered or malicious. Intervals are 95% Wilson intervals.

## Verdicts by assurance level

| level | tampered flagged | of which confident | benign flagged | benign clean | benign inconclusive |
|---|---|---|---|---|---|
| basic | 7/61 (6%-22%) | 0/61 | 0/47 (0%-8%) | 47/47 | 0/47 |
| standard | 59/61 (89%-99%) | 25/61 | 0/47 (0%-8%) | 47/47 | 0/47 |
| thorough | 61/61 (94%-100%) | 28/61 | 0/47 (0%-8%) | 47/47 | 0/47 |
| deep | 55/55 (93%-100%) | 46/55 | 0/34 (0%-10%) | 34/34 | 0/34 |

## Tampering flagged, by threat

| level | threat | flagged |
|---|---|---|
| basic | prompt_injection | 7/25 |
| basic | trained_behaviour | 0/2 |
| basic | weight_implants | 0/34 |
| standard | prompt_injection | 25/25 |
| standard | trained_behaviour | 0/2 |
| standard | weight_implants | 34/34 |
| thorough | prompt_injection | 25/25 |
| thorough | trained_behaviour | 2/2 |
| thorough | weight_implants | 34/34 |
| deep | prompt_injection | 25/25 |
| deep | trained_behaviour | 2/2 |
| deep | weight_implants | 28/28 |

Basic: Of the tampering this level tests (all but trained-in behaviour, which needs behavioural probes), it flagged 7/59 (95% CI 6%-23%).

Standard: Of the tampering this level tests (all but trained-in behaviour, which needs behavioural probes), it flagged 59/59 (95% CI 94%-100%).

## Misses and false alarms

- basic, missed: olmo2-1b/head-boost-s11, olmo2-1b/head-boost-s7, olmo2-1b/head-boost-subtle, olmo2-1b/norm-spike-l3, olmo2-1b/norm-spike-subtle, olmo2-1b/row-implant-s11, olmo2-1b/row-implant-s23, olmo2-1b/row-implant-s7, olmo2-1b/row-implant-subtle, olmo2-1b/template-0-prefix, olmo2-1b/template-1-system, olmo2-1b/template-2-suffix, olmo2-1b/template-4-system, olmo2-1b/tokenizer-normalizer, olmo2-1b/tokenizer-swap, qwen3-0.6b/F1, qwen3-0.6b/F2, qwen3-0.6b/F3, qwen3-0.6b/F3G, qwen3-0.6b/F3G4, qwen3-0.6b/head-boost-s11, qwen3-0.6b/head-boost-s7, qwen3-0.6b/head-boost-subtle, qwen3-0.6b/norm-spike-l3, qwen3-0.6b/norm-spike-subtle, qwen3-0.6b/row-implant-s11, qwen3-0.6b/row-implant-s23, qwen3-0.6b/row-implant-s7, qwen3-0.6b/row-implant-s7-Q4_0, qwen3-0.6b/row-implant-s7-Q8_0, qwen3-0.6b/row-implant-subtle, qwen3-0.6b/template-0-prefix, qwen3-0.6b/template-1-system, qwen3-0.6b/template-2-suffix, qwen3-0.6b/template-4-system, qwen3-0.6b/tokenizer-normalizer, qwen3-0.6b/tokenizer-swap, smollm2-360m/head-boost-s11, smollm2-360m/head-boost-s7, smollm2-360m/head-boost-subtle, smollm2-360m/norm-spike-l3, smollm2-360m/norm-spike-subtle, smollm2-360m/row-implant-s11, smollm2-360m/row-implant-s23, smollm2-360m/row-implant-s7, smollm2-360m/row-implant-s7-Q4_0, smollm2-360m/row-implant-s7-Q8_0, smollm2-360m/row-implant-subtle, smollm2-360m/template-0-prefix, smollm2-360m/template-1-system, smollm2-360m/template-2-suffix, smollm2-360m/template-4-system, smollm2-360m/tokenizer-normalizer, smollm2-360m/tokenizer-swap
- standard, missed: qwen3-0.6b/F1, qwen3-0.6b/F2

## Scans that reached a lower level than asked

- asked deep, reached thorough: qwen3-0.6b/B2G, qwen3-0.6b/B2G4, qwen3-0.6b/F3G, qwen3-0.6b/F3G4, qwen3-0.6b/bartowski-Q4_K_M, qwen3-0.6b/bartowski-Q6_K, qwen3-0.6b/bartowski-Q8_0, qwen3-0.6b/gguf-Q4_0, qwen3-0.6b/gguf-Q8_0, qwen3-0.6b/row-implant-s7-Q4_0, qwen3-0.6b/row-implant-s7-Q8_0, qwen3-0.6b/unsloth-Q4_K_M, qwen3-0.6b/unsloth-Q8_0, smollm2-360m/bartowski-Q4_K_M, smollm2-360m/bartowski-Q8_0, smollm2-360m/gguf-Q4_0, smollm2-360m/gguf-Q8_0, smollm2-360m/row-implant-s7-Q4_0, smollm2-360m/row-implant-s7-Q8_0

## Measured noise floors

| format | floor | benign conversions | worst p99 row error |
|---|---|---|---|
| gguf:Q4_0 | 0.191 | 3 | 0.127 |
| gguf:Q4_K | 0.129 | 1 | 0.0859 |
| gguf:Q6_K | 0.0351 | 3 | 0.0234 |
| gguf:Q8_0 | 0.0121 | 7 | 0.00807 |

## Finding confidence

Measured where the corpus has at least 10 findings of a kind; the rest keep the check's own estimate.

| check · severity | right | wrong | confidence (95% lower bound) | used |
|---|---|---|---|---|
| blackbox.probes · critical | 1 | 0 | 0.21 | check's estimate |
| blackbox.probes · high | 1 | 0 | 0.21 | check's estimate |
| blackbox.probes · low | 2 | 2 | 0.15 | check's estimate |
| provenance.hub_metadata · low | 0 | 1 | 0.00 | check's estimate |
| static.chat_template · high | 13 | 0 | 0.77 | measured |
| static.chat_template · low | 16 | 3 | 0.62 | measured |
| static.chat_template · medium | 5 | 0 | 0.57 | check's estimate |
| static.config_drift · low | 0 | 8 | 0.00 | check's estimate |
| static.config_drift · medium | 6 | 0 | 0.61 | check's estimate |
| static.embedding_compare · low | 0 | 6 | 0.00 | check's estimate |
| static.embedding_compare · medium | 19 | 0 | 0.83 | measured |
| static.inventory · low | 0 | 11 | 0.00 | measured |
| static.norm_compare · medium | 6 | 0 | 0.61 | check's estimate |
| static.pickle_payload · low | 0 | 11 | 0.00 | measured |
| static.tokenizer_integrity · high | 10 | 0 | 0.72 | measured |
| static.tokenizer_integrity · low | 0 | 6 | 0.00 | check's estimate |
| static.tokenizer_integrity · medium | 10 | 0 | 0.72 | measured |
| static.unembedding_compare · low | 0 | 3 | 0.00 | check's estimate |
| static.unembedding_compare · medium | 12 | 0 | 0.76 | measured |
| static.weight_stats · low | 1 | 1 | 0.09 | check's estimate |
| whitebox.unembed_anomaly · low | 0 | 1 | 0.00 | check's estimate |

## Every scan

| item | label | asked | reached | verdict | seconds |
|---|---|---|---|---|---|
| olmo2-1b/cast-f16 | benign | basic | basic | clean | 10.3 |
| olmo2-1b/cast-f16 | benign | standard | standard | clean | 20.5 |
| olmo2-1b/cast-f16 | benign | thorough | thorough | clean | 11.4 |
| olmo2-1b/cast-f16 | benign | deep | deep | clean | 11.4 |
| olmo2-1b/cast-f32 | benign | basic | basic | clean | 10.1 |
| olmo2-1b/cast-f32 | benign | standard | standard | clean | 19.1 |
| olmo2-1b/cast-f32 | benign | thorough | thorough | clean | 12.2 |
| olmo2-1b/cast-f32 | benign | deep | deep | clean | 10.9 |
| olmo2-1b/config | benign | basic | basic | clean | 10.1 |
| olmo2-1b/config | benign | standard | standard | clean | 20.3 |
| olmo2-1b/config | benign | thorough | thorough | clean | 11.2 |
| olmo2-1b/config | benign | deep | deep | clean | 11.1 |
| olmo2-1b/drift-heavy | benign | basic | basic | clean | 11.0 |
| olmo2-1b/drift-heavy | benign | standard | standard | clean | 20.3 |
| olmo2-1b/drift-heavy | benign | thorough | thorough | clean | 12.0 |
| olmo2-1b/drift-heavy | benign | deep | deep | clean | 11.6 |
| olmo2-1b/drift-light | benign | basic | basic | clean | 10.5 |
| olmo2-1b/drift-light | benign | standard | standard | clean | 21.1 |
| olmo2-1b/drift-light | benign | thorough | thorough | clean | 12.0 |
| olmo2-1b/drift-light | benign | deep | deep | clean | 11.6 |
| olmo2-1b/head-boost-s11 | tampered | basic | basic | clean | 10.5 |
| olmo2-1b/head-boost-s11 | tampered | standard | standard | suspicious | 21.3 |
| olmo2-1b/head-boost-s11 | tampered | thorough | thorough | suspicious | 11.8 |
| olmo2-1b/head-boost-s11 | tampered | deep | deep | likely_tampered | 11.9 |
| olmo2-1b/head-boost-s7 | tampered | basic | basic | clean | 11.1 |
| olmo2-1b/head-boost-s7 | tampered | standard | standard | suspicious | 21.9 |
| olmo2-1b/head-boost-s7 | tampered | thorough | thorough | suspicious | 11.6 |
| olmo2-1b/head-boost-s7 | tampered | deep | deep | likely_tampered | 11.8 |
| olmo2-1b/head-boost-subtle | tampered | basic | basic | clean | 11.0 |
| olmo2-1b/head-boost-subtle | tampered | standard | standard | suspicious | 21.9 |
| olmo2-1b/head-boost-subtle | tampered | thorough | thorough | suspicious | 11.7 |
| olmo2-1b/head-boost-subtle | tampered | deep | deep | likely_tampered | 11.9 |
| olmo2-1b/norm-spike-l3 | tampered | basic | basic | clean | 11.2 |
| olmo2-1b/norm-spike-l3 | tampered | standard | standard | suspicious | 21.0 |
| olmo2-1b/norm-spike-l3 | tampered | thorough | thorough | suspicious | 11.5 |
| olmo2-1b/norm-spike-l3 | tampered | deep | deep | suspicious | 11.5 |
| olmo2-1b/norm-spike-subtle | tampered | basic | basic | clean | 10.8 |
| olmo2-1b/norm-spike-subtle | tampered | standard | standard | suspicious | 20.2 |
| olmo2-1b/norm-spike-subtle | tampered | thorough | thorough | suspicious | 11.5 |
| olmo2-1b/norm-spike-subtle | tampered | deep | deep | suspicious | 11.6 |
| olmo2-1b/reshard-2 | benign | basic | basic | clean | 10.8 |
| olmo2-1b/reshard-2 | benign | standard | standard | clean | 20.8 |
| olmo2-1b/reshard-2 | benign | thorough | thorough | clean | 11.8 |
| olmo2-1b/reshard-2 | benign | deep | deep | clean | 11.7 |
| olmo2-1b/reshard-3 | benign | basic | basic | clean | 11.2 |
| olmo2-1b/reshard-3 | benign | standard | standard | clean | 20.9 |
| olmo2-1b/reshard-3 | benign | thorough | thorough | clean | 11.6 |
| olmo2-1b/reshard-3 | benign | deep | deep | clean | 11.6 |
| olmo2-1b/row-implant-s11 | tampered | basic | basic | clean | 10.5 |
| olmo2-1b/row-implant-s11 | tampered | standard | standard | suspicious | 20.6 |
| olmo2-1b/row-implant-s11 | tampered | thorough | thorough | suspicious | 11.5 |
| olmo2-1b/row-implant-s11 | tampered | deep | deep | likely_tampered | 11.8 |
| olmo2-1b/row-implant-s23 | tampered | basic | basic | clean | 10.9 |
| olmo2-1b/row-implant-s23 | tampered | standard | standard | suspicious | 22.1 |
| olmo2-1b/row-implant-s23 | tampered | thorough | thorough | suspicious | 11.8 |
| olmo2-1b/row-implant-s23 | tampered | deep | deep | likely_tampered | 11.8 |
| olmo2-1b/row-implant-s7 | tampered | basic | basic | clean | 10.3 |
| olmo2-1b/row-implant-s7 | tampered | standard | standard | suspicious | 21.8 |
| olmo2-1b/row-implant-s7 | tampered | thorough | thorough | suspicious | 11.5 |
| olmo2-1b/row-implant-s7 | tampered | deep | deep | likely_tampered | 12.4 |
| olmo2-1b/row-implant-subtle | tampered | basic | basic | clean | 12.1 |
| olmo2-1b/row-implant-subtle | tampered | standard | standard | suspicious | 22.9 |
| olmo2-1b/row-implant-subtle | tampered | thorough | thorough | suspicious | 11.6 |
| olmo2-1b/row-implant-subtle | tampered | deep | deep | likely_tampered | 11.7 |
| olmo2-1b/template-0-prefix | tampered | basic | basic | clean | 10.1 |
| olmo2-1b/template-0-prefix | tampered | standard | standard | likely_tampered | 20.2 |
| olmo2-1b/template-0-prefix | tampered | thorough | thorough | likely_tampered | 11.4 |
| olmo2-1b/template-0-prefix | tampered | deep | deep | likely_tampered | 11.1 |
| olmo2-1b/template-1-system | tampered | basic | basic | clean | 9.9 |
| olmo2-1b/template-1-system | tampered | standard | standard | likely_tampered | 19.8 |
| olmo2-1b/template-1-system | tampered | thorough | thorough | likely_tampered | 11.3 |
| olmo2-1b/template-1-system | tampered | deep | deep | likely_tampered | 11.1 |
| olmo2-1b/template-2-suffix | tampered | basic | basic | clean | 9.8 |
| olmo2-1b/template-2-suffix | tampered | standard | standard | likely_tampered | 19.6 |
| olmo2-1b/template-2-suffix | tampered | thorough | thorough | likely_tampered | 11.2 |
| olmo2-1b/template-2-suffix | tampered | deep | deep | likely_tampered | 11.0 |
| olmo2-1b/template-3-prefix | tampered | basic | basic | suspicious | 9.9 |
| olmo2-1b/template-3-prefix | tampered | standard | standard | likely_tampered | 19.9 |
| olmo2-1b/template-3-prefix | tampered | thorough | thorough | likely_tampered | 11.4 |
| olmo2-1b/template-3-prefix | tampered | deep | deep | likely_tampered | 11.1 |
| olmo2-1b/template-4-system | tampered | basic | basic | clean | 10.0 |
| olmo2-1b/template-4-system | tampered | standard | standard | likely_tampered | 20.1 |
| olmo2-1b/template-4-system | tampered | thorough | thorough | likely_tampered | 11.1 |
| olmo2-1b/template-4-system | tampered | deep | deep | likely_tampered | 11.2 |
| olmo2-1b/template-comment | benign | basic | basic | clean | 10.0 |
| olmo2-1b/template-comment | benign | standard | standard | clean | 19.9 |
| olmo2-1b/template-comment | benign | thorough | thorough | clean | 11.4 |
| olmo2-1b/template-comment | benign | deep | deep | clean | 11.1 |
| olmo2-1b/template-whitespace | benign | basic | basic | clean | 9.9 |
| olmo2-1b/template-whitespace | benign | standard | standard | clean | 20.2 |
| olmo2-1b/template-whitespace | benign | thorough | thorough | clean | 11.4 |
| olmo2-1b/template-whitespace | benign | deep | deep | clean | 11.2 |
| olmo2-1b/tokenizer-normalizer | tampered | basic | basic | clean | 9.9 |
| olmo2-1b/tokenizer-normalizer | tampered | standard | standard | likely_tampered | 19.9 |
| olmo2-1b/tokenizer-normalizer | tampered | thorough | thorough | likely_tampered | 11.3 |
| olmo2-1b/tokenizer-normalizer | tampered | deep | deep | likely_tampered | 11.3 |
| olmo2-1b/tokenizer-swap | tampered | basic | basic | clean | 9.6 |
| olmo2-1b/tokenizer-swap | tampered | standard | standard | suspicious | 19.9 |
| olmo2-1b/tokenizer-swap | tampered | thorough | thorough | suspicious | 11.3 |
| olmo2-1b/tokenizer-swap | tampered | deep | deep | suspicious | 11.3 |
| olmo2-1b/tokenizer-zero-width | tampered | basic | basic | suspicious | 10.0 |
| olmo2-1b/tokenizer-zero-width | tampered | standard | standard | likely_tampered | 20.5 |
| olmo2-1b/tokenizer-zero-width | tampered | thorough | thorough | likely_tampered | 11.3 |
| olmo2-1b/tokenizer-zero-width | tampered | deep | deep | likely_tampered | 11.2 |
| olmo2-1b/unsloth | presumed_benign | basic | basic | clean | 10.1 |
| olmo2-1b/unsloth | presumed_benign | standard | standard | clean | 16.7 |
| olmo2-1b/unsloth | presumed_benign | thorough | thorough | clean | 11.4 |
| olmo2-1b/unsloth | presumed_benign | deep | deep | clean | 11.4 |
| qwen3-0.6b/B1 | benign | basic | basic | clean | 4.1 |
| qwen3-0.6b/B1 | benign | standard | standard | clean | 10.8 |
| qwen3-0.6b/B1 | benign | thorough | thorough | clean | 6.9 |
| qwen3-0.6b/B1 | benign | deep | deep | clean | 7.1 |
| qwen3-0.6b/B2 | benign | basic | basic | clean | 5.5 |
| qwen3-0.6b/B2 | benign | standard | standard | clean | 10.5 |
| qwen3-0.6b/B2 | benign | thorough | thorough | clean | 7.9 |
| qwen3-0.6b/B2 | benign | deep | deep | clean | 8.0 |
| qwen3-0.6b/B2G | benign | basic | basic | clean | 4.2 |
| qwen3-0.6b/B2G | benign | standard | standard | clean | 10.9 |
| qwen3-0.6b/B2G | benign | thorough | thorough | clean | 8.8 |
| qwen3-0.6b/B2G | benign | deep | thorough | clean | 36.4 |
| qwen3-0.6b/B2G4 | benign | basic | basic | clean | 4.0 |
| qwen3-0.6b/B2G4 | benign | standard | standard | clean | 11.8 |
| qwen3-0.6b/B2G4 | benign | thorough | thorough | clean | 8.8 |
| qwen3-0.6b/B2G4 | benign | deep | thorough | clean | 36.3 |
| qwen3-0.6b/F1 | tampered | basic | basic | clean | 3.9 |
| qwen3-0.6b/F1 | tampered | standard | standard | clean | 11.6 |
| qwen3-0.6b/F1 | tampered | thorough | thorough | likely_tampered | 7.0 |
| qwen3-0.6b/F1 | tampered | deep | deep | likely_tampered | 7.2 |
| qwen3-0.6b/F2 | tampered | basic | basic | clean | 3.8 |
| qwen3-0.6b/F2 | tampered | standard | standard | clean | 11.6 |
| qwen3-0.6b/F2 | tampered | thorough | thorough | malicious | 7.0 |
| qwen3-0.6b/F2 | tampered | deep | deep | malicious | 7.0 |
| qwen3-0.6b/F3 | tampered | basic | basic | clean | 5.7 |
| qwen3-0.6b/F3 | tampered | standard | standard | likely_tampered | 12.0 |
| qwen3-0.6b/F3 | tampered | thorough | thorough | likely_tampered | 8.2 |
| qwen3-0.6b/F3 | tampered | deep | deep | likely_tampered | 8.2 |
| qwen3-0.6b/F3G | tampered | basic | basic | clean | 4.4 |
| qwen3-0.6b/F3G | tampered | standard | standard | likely_tampered | 11.9 |
| qwen3-0.6b/F3G | tampered | thorough | thorough | likely_tampered | 9.2 |
| qwen3-0.6b/F3G | tampered | deep | thorough | likely_tampered | 36.6 |
| qwen3-0.6b/F3G4 | tampered | basic | basic | clean | 4.1 |
| qwen3-0.6b/F3G4 | tampered | standard | standard | likely_tampered | 12.3 |
| qwen3-0.6b/F3G4 | tampered | thorough | thorough | likely_tampered | 9.1 |
| qwen3-0.6b/F3G4 | tampered | deep | thorough | likely_tampered | 36.7 |
| qwen3-0.6b/F4 | tampered | basic | basic | suspicious | 5.6 |
| qwen3-0.6b/F4 | tampered | standard | standard | likely_tampered | 10.8 |
| qwen3-0.6b/F4 | tampered | thorough | thorough | likely_tampered | 7.8 |
| qwen3-0.6b/F4 | tampered | deep | deep | likely_tampered | 7.8 |
| qwen3-0.6b/bartowski-Q4_K_M | presumed_benign | basic | basic | clean | 4.1 |
| qwen3-0.6b/bartowski-Q4_K_M | presumed_benign | standard | standard | clean | 10.5 |
| qwen3-0.6b/bartowski-Q4_K_M | presumed_benign | thorough | thorough | clean | 8.7 |
| qwen3-0.6b/bartowski-Q4_K_M | presumed_benign | deep | thorough | clean | 33.4 |
| qwen3-0.6b/bartowski-Q6_K | presumed_benign | basic | basic | clean | 4.0 |
| qwen3-0.6b/bartowski-Q6_K | presumed_benign | standard | standard | clean | 10.0 |
| qwen3-0.6b/bartowski-Q6_K | presumed_benign | thorough | thorough | clean | 8.8 |
| qwen3-0.6b/bartowski-Q6_K | presumed_benign | deep | thorough | clean | 33.7 |
| qwen3-0.6b/bartowski-Q8_0 | presumed_benign | basic | basic | clean | 4.0 |
| qwen3-0.6b/bartowski-Q8_0 | presumed_benign | standard | standard | clean | 10.3 |
| qwen3-0.6b/bartowski-Q8_0 | presumed_benign | thorough | thorough | clean | 8.6 |
| qwen3-0.6b/bartowski-Q8_0 | presumed_benign | deep | thorough | clean | 33.3 |
| qwen3-0.6b/cast-f16 | benign | basic | basic | clean | 4.9 |
| qwen3-0.6b/cast-f16 | benign | standard | standard | clean | 11.8 |
| qwen3-0.6b/cast-f16 | benign | thorough | thorough | clean | 7.7 |
| qwen3-0.6b/cast-f16 | benign | deep | deep | clean | 7.9 |
| qwen3-0.6b/cast-f32 | benign | basic | basic | clean | 5.0 |
| qwen3-0.6b/cast-f32 | benign | standard | standard | clean | 11.6 |
| qwen3-0.6b/cast-f32 | benign | thorough | thorough | clean | 7.4 |
| qwen3-0.6b/cast-f32 | benign | deep | deep | clean | 7.6 |
| qwen3-0.6b/config | benign | basic | basic | clean | 5.5 |
| qwen3-0.6b/config | benign | standard | standard | clean | 12.7 |
| qwen3-0.6b/config | benign | thorough | thorough | clean | 7.8 |
| qwen3-0.6b/config | benign | deep | deep | clean | 7.9 |
| qwen3-0.6b/drift-heavy | benign | basic | basic | clean | 5.4 |
| qwen3-0.6b/drift-heavy | benign | standard | standard | clean | 12.4 |
| qwen3-0.6b/drift-heavy | benign | thorough | thorough | clean | 7.9 |
| qwen3-0.6b/drift-heavy | benign | deep | deep | clean | 8.0 |
| qwen3-0.6b/drift-light | benign | basic | basic | clean | 5.4 |
| qwen3-0.6b/drift-light | benign | standard | standard | clean | 12.8 |
| qwen3-0.6b/drift-light | benign | thorough | thorough | clean | 8.0 |
| qwen3-0.6b/drift-light | benign | deep | deep | clean | 8.0 |
| qwen3-0.6b/gguf-Q4_0 | benign | basic | basic | clean | 4.1 |
| qwen3-0.6b/gguf-Q4_0 | benign | standard | standard | clean | 11.3 |
| qwen3-0.6b/gguf-Q4_0 | benign | thorough | thorough | clean | 9.0 |
| qwen3-0.6b/gguf-Q4_0 | benign | deep | thorough | clean | 36.7 |
| qwen3-0.6b/gguf-Q8_0 | benign | basic | basic | clean | 4.5 |
| qwen3-0.6b/gguf-Q8_0 | benign | standard | standard | clean | 11.0 |
| qwen3-0.6b/gguf-Q8_0 | benign | thorough | thorough | clean | 9.0 |
| qwen3-0.6b/gguf-Q8_0 | benign | deep | thorough | clean | 37.0 |
| qwen3-0.6b/head-boost-s11 | tampered | basic | basic | clean | 5.6 |
| qwen3-0.6b/head-boost-s11 | tampered | standard | standard | suspicious | 12.9 |
| qwen3-0.6b/head-boost-s11 | tampered | thorough | thorough | suspicious | 8.3 |
| qwen3-0.6b/head-boost-s11 | tampered | deep | deep | likely_tampered | 8.5 |
| qwen3-0.6b/head-boost-s7 | tampered | basic | basic | clean | 5.4 |
| qwen3-0.6b/head-boost-s7 | tampered | standard | standard | suspicious | 12.6 |
| qwen3-0.6b/head-boost-s7 | tampered | thorough | thorough | suspicious | 8.5 |
| qwen3-0.6b/head-boost-s7 | tampered | deep | deep | likely_tampered | 8.2 |
| qwen3-0.6b/head-boost-subtle | tampered | basic | basic | clean | 5.3 |
| qwen3-0.6b/head-boost-subtle | tampered | standard | standard | suspicious | 12.9 |
| qwen3-0.6b/head-boost-subtle | tampered | thorough | thorough | suspicious | 7.8 |
| qwen3-0.6b/head-boost-subtle | tampered | deep | deep | likely_tampered | 8.5 |
| qwen3-0.6b/mlx-bf16 | presumed_benign | basic | basic | clean | 3.6 |
| qwen3-0.6b/mlx-bf16 | presumed_benign | standard | standard | clean | 8.3 |
| qwen3-0.6b/mlx-bf16 | presumed_benign | thorough | thorough | clean | 6.9 |
| qwen3-0.6b/mlx-bf16 | presumed_benign | deep | deep | clean | 6.9 |
| qwen3-0.6b/norm-spike-l3 | tampered | basic | basic | clean | 5.5 |
| qwen3-0.6b/norm-spike-l3 | tampered | standard | standard | suspicious | 12.7 |
| qwen3-0.6b/norm-spike-l3 | tampered | thorough | thorough | suspicious | 7.9 |
| qwen3-0.6b/norm-spike-l3 | tampered | deep | deep | suspicious | 8.0 |
| qwen3-0.6b/norm-spike-subtle | tampered | basic | basic | clean | 5.4 |
| qwen3-0.6b/norm-spike-subtle | tampered | standard | standard | suspicious | 12.6 |
| qwen3-0.6b/norm-spike-subtle | tampered | thorough | thorough | suspicious | 8.0 |
| qwen3-0.6b/norm-spike-subtle | tampered | deep | deep | suspicious | 8.1 |
| qwen3-0.6b/reshard-2 | benign | basic | basic | clean | 5.5 |
| qwen3-0.6b/reshard-2 | benign | standard | standard | clean | 11.8 |
| qwen3-0.6b/reshard-2 | benign | thorough | thorough | clean | 7.9 |
| qwen3-0.6b/reshard-2 | benign | deep | deep | clean | 8.0 |
| qwen3-0.6b/reshard-3 | benign | basic | basic | clean | 5.4 |
| qwen3-0.6b/reshard-3 | benign | standard | standard | clean | 10.7 |
| qwen3-0.6b/reshard-3 | benign | thorough | thorough | clean | 7.8 |
| qwen3-0.6b/reshard-3 | benign | deep | deep | clean | 7.9 |
| qwen3-0.6b/row-implant-s11 | tampered | basic | basic | clean | 6.0 |
| qwen3-0.6b/row-implant-s11 | tampered | standard | standard | suspicious | 13.7 |
| qwen3-0.6b/row-implant-s11 | tampered | thorough | thorough | suspicious | 7.6 |
| qwen3-0.6b/row-implant-s11 | tampered | deep | deep | likely_tampered | 7.7 |
| qwen3-0.6b/row-implant-s23 | tampered | basic | basic | clean | 5.5 |
| qwen3-0.6b/row-implant-s23 | tampered | standard | standard | suspicious | 13.0 |
| qwen3-0.6b/row-implant-s23 | tampered | thorough | thorough | suspicious | 7.9 |
| qwen3-0.6b/row-implant-s23 | tampered | deep | deep | likely_tampered | 7.6 |
| qwen3-0.6b/row-implant-s7 | tampered | basic | basic | clean | 6.0 |
| qwen3-0.6b/row-implant-s7 | tampered | standard | standard | suspicious | 13.7 |
| qwen3-0.6b/row-implant-s7 | tampered | thorough | thorough | suspicious | 7.8 |
| qwen3-0.6b/row-implant-s7 | tampered | deep | deep | likely_tampered | 7.7 |
| qwen3-0.6b/row-implant-s7-Q4_0 | tampered | basic | basic | clean | 4.2 |
| qwen3-0.6b/row-implant-s7-Q4_0 | tampered | standard | standard | suspicious | 11.5 |
| qwen3-0.6b/row-implant-s7-Q4_0 | tampered | thorough | thorough | suspicious | 8.6 |
| qwen3-0.6b/row-implant-s7-Q4_0 | tampered | deep | thorough | suspicious | 36.4 |
| qwen3-0.6b/row-implant-s7-Q8_0 | tampered | basic | basic | clean | 4.4 |
| qwen3-0.6b/row-implant-s7-Q8_0 | tampered | standard | standard | suspicious | 10.9 |
| qwen3-0.6b/row-implant-s7-Q8_0 | tampered | thorough | thorough | suspicious | 8.9 |
| qwen3-0.6b/row-implant-s7-Q8_0 | tampered | deep | thorough | suspicious | 38.7 |
| qwen3-0.6b/row-implant-subtle | tampered | basic | basic | clean | 5.4 |
| qwen3-0.6b/row-implant-subtle | tampered | standard | standard | suspicious | 13.0 |
| qwen3-0.6b/row-implant-subtle | tampered | thorough | thorough | suspicious | 7.8 |
| qwen3-0.6b/row-implant-subtle | tampered | deep | deep | likely_tampered | 8.3 |
| qwen3-0.6b/template-0-prefix | tampered | basic | basic | clean | 5.3 |
| qwen3-0.6b/template-0-prefix | tampered | standard | standard | likely_tampered | 13.0 |
| qwen3-0.6b/template-0-prefix | tampered | thorough | thorough | likely_tampered | 7.8 |
| qwen3-0.6b/template-0-prefix | tampered | deep | deep | likely_tampered | 8.1 |
| qwen3-0.6b/template-1-system | tampered | basic | basic | clean | 5.5 |
| qwen3-0.6b/template-1-system | tampered | standard | standard | likely_tampered | 12.9 |
| qwen3-0.6b/template-1-system | tampered | thorough | thorough | likely_tampered | 7.9 |
| qwen3-0.6b/template-1-system | tampered | deep | deep | likely_tampered | 8.1 |
| qwen3-0.6b/template-2-suffix | tampered | basic | basic | clean | 5.5 |
| qwen3-0.6b/template-2-suffix | tampered | standard | standard | likely_tampered | 12.8 |
| qwen3-0.6b/template-2-suffix | tampered | thorough | thorough | likely_tampered | 7.8 |
| qwen3-0.6b/template-2-suffix | tampered | deep | deep | likely_tampered | 7.8 |
| qwen3-0.6b/template-3-prefix | tampered | basic | basic | suspicious | 5.5 |
| qwen3-0.6b/template-3-prefix | tampered | standard | standard | likely_tampered | 12.6 |
| qwen3-0.6b/template-3-prefix | tampered | thorough | thorough | likely_tampered | 7.8 |
| qwen3-0.6b/template-3-prefix | tampered | deep | deep | likely_tampered | 8.0 |
| qwen3-0.6b/template-4-system | tampered | basic | basic | clean | 5.6 |
| qwen3-0.6b/template-4-system | tampered | standard | standard | likely_tampered | 13.0 |
| qwen3-0.6b/template-4-system | tampered | thorough | thorough | likely_tampered | 7.7 |
| qwen3-0.6b/template-4-system | tampered | deep | deep | likely_tampered | 8.0 |
| qwen3-0.6b/template-comment | benign | basic | basic | clean | 5.6 |
| qwen3-0.6b/template-comment | benign | standard | standard | clean | 12.3 |
| qwen3-0.6b/template-comment | benign | thorough | thorough | clean | 7.9 |
| qwen3-0.6b/template-comment | benign | deep | deep | clean | 8.1 |
| qwen3-0.6b/template-whitespace | benign | basic | basic | clean | 5.5 |
| qwen3-0.6b/template-whitespace | benign | standard | standard | clean | 13.0 |
| qwen3-0.6b/template-whitespace | benign | thorough | thorough | clean | 7.9 |
| qwen3-0.6b/template-whitespace | benign | deep | deep | clean | 7.9 |
| qwen3-0.6b/tokenizer-normalizer | tampered | basic | basic | clean | 5.3 |
| qwen3-0.6b/tokenizer-normalizer | tampered | standard | standard | likely_tampered | 12.8 |
| qwen3-0.6b/tokenizer-normalizer | tampered | thorough | thorough | likely_tampered | 7.8 |
| qwen3-0.6b/tokenizer-normalizer | tampered | deep | deep | likely_tampered | 7.9 |
| qwen3-0.6b/tokenizer-swap | tampered | basic | basic | clean | 5.4 |
| qwen3-0.6b/tokenizer-swap | tampered | standard | standard | suspicious | 11.6 |
| qwen3-0.6b/tokenizer-swap | tampered | thorough | thorough | suspicious | 7.8 |
| qwen3-0.6b/tokenizer-swap | tampered | deep | deep | suspicious | 8.0 |
| qwen3-0.6b/tokenizer-zero-width | tampered | basic | basic | suspicious | 4.9 |
| qwen3-0.6b/tokenizer-zero-width | tampered | standard | standard | likely_tampered | 10.6 |
| qwen3-0.6b/tokenizer-zero-width | tampered | thorough | thorough | likely_tampered | 7.8 |
| qwen3-0.6b/tokenizer-zero-width | tampered | deep | deep | likely_tampered | 8.0 |
| qwen3-0.6b/unsloth | presumed_benign | basic | basic | clean | 3.6 |
| qwen3-0.6b/unsloth | presumed_benign | standard | standard | clean | 10.1 |
| qwen3-0.6b/unsloth | presumed_benign | thorough | thorough | clean | 6.9 |
| qwen3-0.6b/unsloth | presumed_benign | deep | deep | clean | 6.9 |
| qwen3-0.6b/unsloth-Q4_K_M | presumed_benign | basic | basic | clean | 3.9 |
| qwen3-0.6b/unsloth-Q4_K_M | presumed_benign | standard | standard | clean | 10.5 |
| qwen3-0.6b/unsloth-Q4_K_M | presumed_benign | thorough | thorough | clean | 8.6 |
| qwen3-0.6b/unsloth-Q4_K_M | presumed_benign | deep | thorough | clean | 33.1 |
| qwen3-0.6b/unsloth-Q8_0 | presumed_benign | basic | basic | clean | 4.0 |
| qwen3-0.6b/unsloth-Q8_0 | presumed_benign | standard | standard | clean | 11.8 |
| qwen3-0.6b/unsloth-Q8_0 | presumed_benign | thorough | thorough | clean | 8.4 |
| qwen3-0.6b/unsloth-Q8_0 | presumed_benign | deep | thorough | clean | 33.3 |
| qwen3-1.7b/unsloth | presumed_benign | basic | basic | clean | 10.2 |
| qwen3-1.7b/unsloth | presumed_benign | standard | standard | clean | 16.2 |
| qwen3-1.7b/unsloth | presumed_benign | thorough | thorough | clean | 14.8 |
| qwen3-1.7b/unsloth | presumed_benign | deep | deep | clean | 14.8 |
| smollm2-360m/bartowski-Q4_K_M | presumed_benign | basic | basic | clean | 1.3 |
| smollm2-360m/bartowski-Q4_K_M | presumed_benign | standard | standard | clean | 2.5 |
| smollm2-360m/bartowski-Q4_K_M | presumed_benign | thorough | thorough | clean | 2.9 |
| smollm2-360m/bartowski-Q4_K_M | presumed_benign | deep | thorough | clean | 27.2 |
| smollm2-360m/bartowski-Q8_0 | presumed_benign | basic | basic | clean | 1.3 |
| smollm2-360m/bartowski-Q8_0 | presumed_benign | standard | standard | clean | 2.8 |
| smollm2-360m/bartowski-Q8_0 | presumed_benign | thorough | thorough | clean | 2.9 |
| smollm2-360m/bartowski-Q8_0 | presumed_benign | deep | thorough | clean | 27.0 |
| smollm2-360m/cast-f16 | benign | basic | basic | clean | 1.4 |
| smollm2-360m/cast-f16 | benign | standard | standard | clean | 2.5 |
| smollm2-360m/cast-f16 | benign | thorough | thorough | clean | 2.7 |
| smollm2-360m/cast-f16 | benign | deep | deep | clean | 2.9 |
| smollm2-360m/cast-f32 | benign | basic | basic | clean | 1.6 |
| smollm2-360m/cast-f32 | benign | standard | standard | clean | 2.7 |
| smollm2-360m/cast-f32 | benign | thorough | thorough | clean | 2.7 |
| smollm2-360m/cast-f32 | benign | deep | deep | clean | 2.9 |
| smollm2-360m/config | benign | basic | basic | clean | 1.5 |
| smollm2-360m/config | benign | standard | standard | clean | 2.8 |
| smollm2-360m/config | benign | thorough | thorough | clean | 2.6 |
| smollm2-360m/config | benign | deep | deep | clean | 2.6 |
| smollm2-360m/drift-heavy | benign | basic | basic | clean | 1.5 |
| smollm2-360m/drift-heavy | benign | standard | standard | clean | 2.7 |
| smollm2-360m/drift-heavy | benign | thorough | thorough | clean | 2.8 |
| smollm2-360m/drift-heavy | benign | deep | deep | clean | 2.8 |
| smollm2-360m/drift-light | benign | basic | basic | clean | 1.6 |
| smollm2-360m/drift-light | benign | standard | standard | clean | 2.9 |
| smollm2-360m/drift-light | benign | thorough | thorough | clean | 2.7 |
| smollm2-360m/drift-light | benign | deep | deep | clean | 2.7 |
| smollm2-360m/gguf-Q4_0 | benign | basic | basic | clean | 1.3 |
| smollm2-360m/gguf-Q4_0 | benign | standard | standard | clean | 2.6 |
| smollm2-360m/gguf-Q4_0 | benign | thorough | thorough | clean | 2.9 |
| smollm2-360m/gguf-Q4_0 | benign | deep | thorough | clean | 29.7 |
| smollm2-360m/gguf-Q8_0 | benign | basic | basic | clean | 1.5 |
| smollm2-360m/gguf-Q8_0 | benign | standard | standard | clean | 2.5 |
| smollm2-360m/gguf-Q8_0 | benign | thorough | thorough | clean | 2.9 |
| smollm2-360m/gguf-Q8_0 | benign | deep | thorough | clean | 29.7 |
| smollm2-360m/head-boost-s11 | tampered | basic | basic | clean | 2.4 |
| smollm2-360m/head-boost-s11 | tampered | standard | standard | suspicious | 4.3 |
| smollm2-360m/head-boost-s11 | tampered | thorough | thorough | suspicious | 4.0 |
| smollm2-360m/head-boost-s11 | tampered | deep | deep | likely_tampered | 3.9 |
| smollm2-360m/head-boost-s7 | tampered | basic | basic | clean | 2.2 |
| smollm2-360m/head-boost-s7 | tampered | standard | standard | suspicious | 4.3 |
| smollm2-360m/head-boost-s7 | tampered | thorough | thorough | suspicious | 3.5 |
| smollm2-360m/head-boost-s7 | tampered | deep | deep | likely_tampered | 3.6 |
| smollm2-360m/head-boost-subtle | tampered | basic | basic | clean | 2.1 |
| smollm2-360m/head-boost-subtle | tampered | standard | standard | suspicious | 4.2 |
| smollm2-360m/head-boost-subtle | tampered | thorough | thorough | suspicious | 3.5 |
| smollm2-360m/head-boost-subtle | tampered | deep | deep | likely_tampered | 3.7 |
| smollm2-360m/norm-spike-l3 | tampered | basic | basic | clean | 1.6 |
| smollm2-360m/norm-spike-l3 | tampered | standard | standard | suspicious | 2.7 |
| smollm2-360m/norm-spike-l3 | tampered | thorough | thorough | suspicious | 2.7 |
| smollm2-360m/norm-spike-l3 | tampered | deep | deep | suspicious | 2.8 |
| smollm2-360m/norm-spike-subtle | tampered | basic | basic | clean | 1.5 |
| smollm2-360m/norm-spike-subtle | tampered | standard | standard | suspicious | 2.6 |
| smollm2-360m/norm-spike-subtle | tampered | thorough | thorough | suspicious | 2.7 |
| smollm2-360m/norm-spike-subtle | tampered | deep | deep | suspicious | 2.8 |
| smollm2-360m/reshard-2 | benign | basic | basic | clean | 1.6 |
| smollm2-360m/reshard-2 | benign | standard | standard | clean | 2.6 |
| smollm2-360m/reshard-2 | benign | thorough | thorough | clean | 2.7 |
| smollm2-360m/reshard-2 | benign | deep | deep | clean | 2.8 |
| smollm2-360m/reshard-3 | benign | basic | basic | clean | 1.6 |
| smollm2-360m/reshard-3 | benign | standard | standard | clean | 2.4 |
| smollm2-360m/reshard-3 | benign | thorough | thorough | clean | 2.7 |
| smollm2-360m/reshard-3 | benign | deep | deep | clean | 2.8 |
| smollm2-360m/row-implant-s11 | tampered | basic | basic | clean | 1.7 |
| smollm2-360m/row-implant-s11 | tampered | standard | standard | suspicious | 2.5 |
| smollm2-360m/row-implant-s11 | tampered | thorough | thorough | suspicious | 2.8 |
| smollm2-360m/row-implant-s11 | tampered | deep | deep | likely_tampered | 3.0 |
| smollm2-360m/row-implant-s23 | tampered | basic | basic | clean | 1.3 |
| smollm2-360m/row-implant-s23 | tampered | standard | standard | suspicious | 2.7 |
| smollm2-360m/row-implant-s23 | tampered | thorough | thorough | likely_tampered | 2.7 |
| smollm2-360m/row-implant-s23 | tampered | deep | deep | likely_tampered | 2.9 |
| smollm2-360m/row-implant-s7 | tampered | basic | basic | clean | 1.3 |
| smollm2-360m/row-implant-s7 | tampered | standard | standard | suspicious | 3.0 |
| smollm2-360m/row-implant-s7 | tampered | thorough | thorough | suspicious | 2.9 |
| smollm2-360m/row-implant-s7 | tampered | deep | deep | likely_tampered | 3.2 |
| smollm2-360m/row-implant-s7-Q4_0 | tampered | basic | basic | clean | 1.4 |
| smollm2-360m/row-implant-s7-Q4_0 | tampered | standard | standard | suspicious | 2.7 |
| smollm2-360m/row-implant-s7-Q4_0 | tampered | thorough | thorough | suspicious | 2.6 |
| smollm2-360m/row-implant-s7-Q4_0 | tampered | deep | thorough | suspicious | 29.4 |
| smollm2-360m/row-implant-s7-Q8_0 | tampered | basic | basic | clean | 1.3 |
| smollm2-360m/row-implant-s7-Q8_0 | tampered | standard | standard | suspicious | 2.3 |
| smollm2-360m/row-implant-s7-Q8_0 | tampered | thorough | thorough | suspicious | 2.7 |
| smollm2-360m/row-implant-s7-Q8_0 | tampered | deep | thorough | suspicious | 29.6 |
| smollm2-360m/row-implant-subtle | tampered | basic | basic | clean | 1.7 |
| smollm2-360m/row-implant-subtle | tampered | standard | standard | suspicious | 2.8 |
| smollm2-360m/row-implant-subtle | tampered | thorough | thorough | suspicious | 3.0 |
| smollm2-360m/row-implant-subtle | tampered | deep | deep | likely_tampered | 3.0 |
| smollm2-360m/template-0-prefix | tampered | basic | basic | clean | 1.5 |
| smollm2-360m/template-0-prefix | tampered | standard | standard | likely_tampered | 2.6 |
| smollm2-360m/template-0-prefix | tampered | thorough | thorough | likely_tampered | 2.6 |
| smollm2-360m/template-0-prefix | tampered | deep | deep | likely_tampered | 2.7 |
| smollm2-360m/template-1-system | tampered | basic | basic | clean | 1.5 |
| smollm2-360m/template-1-system | tampered | standard | standard | likely_tampered | 2.7 |
| smollm2-360m/template-1-system | tampered | thorough | thorough | likely_tampered | 2.6 |
| smollm2-360m/template-1-system | tampered | deep | deep | likely_tampered | 2.8 |
| smollm2-360m/template-2-suffix | tampered | basic | basic | clean | 1.5 |
| smollm2-360m/template-2-suffix | tampered | standard | standard | likely_tampered | 2.8 |
| smollm2-360m/template-2-suffix | tampered | thorough | thorough | likely_tampered | 2.5 |
| smollm2-360m/template-2-suffix | tampered | deep | deep | likely_tampered | 2.6 |
| smollm2-360m/template-3-prefix | tampered | basic | basic | suspicious | 1.6 |
| smollm2-360m/template-3-prefix | tampered | standard | standard | likely_tampered | 2.6 |
| smollm2-360m/template-3-prefix | tampered | thorough | thorough | likely_tampered | 2.6 |
| smollm2-360m/template-3-prefix | tampered | deep | deep | likely_tampered | 2.6 |
| smollm2-360m/template-4-system | tampered | basic | basic | clean | 1.5 |
| smollm2-360m/template-4-system | tampered | standard | standard | likely_tampered | 2.6 |
| smollm2-360m/template-4-system | tampered | thorough | thorough | likely_tampered | 2.5 |
| smollm2-360m/template-4-system | tampered | deep | deep | likely_tampered | 2.5 |
| smollm2-360m/template-comment | benign | basic | basic | clean | 1.5 |
| smollm2-360m/template-comment | benign | standard | standard | clean | 2.7 |
| smollm2-360m/template-comment | benign | thorough | thorough | clean | 2.6 |
| smollm2-360m/template-comment | benign | deep | deep | clean | 2.7 |
| smollm2-360m/template-whitespace | benign | basic | basic | clean | 1.5 |
| smollm2-360m/template-whitespace | benign | standard | standard | clean | 2.6 |
| smollm2-360m/template-whitespace | benign | thorough | thorough | clean | 2.7 |
| smollm2-360m/template-whitespace | benign | deep | deep | clean | 2.6 |
| smollm2-360m/tokenizer-normalizer | tampered | basic | basic | clean | 1.4 |
| smollm2-360m/tokenizer-normalizer | tampered | standard | standard | likely_tampered | 2.6 |
| smollm2-360m/tokenizer-normalizer | tampered | thorough | thorough | likely_tampered | 2.6 |
| smollm2-360m/tokenizer-normalizer | tampered | deep | deep | likely_tampered | 2.6 |
| smollm2-360m/tokenizer-swap | tampered | basic | basic | clean | 1.5 |
| smollm2-360m/tokenizer-swap | tampered | standard | standard | suspicious | 2.7 |
| smollm2-360m/tokenizer-swap | tampered | thorough | thorough | suspicious | 2.6 |
| smollm2-360m/tokenizer-swap | tampered | deep | deep | suspicious | 2.6 |
| smollm2-360m/tokenizer-zero-width | tampered | basic | basic | suspicious | 1.5 |
| smollm2-360m/tokenizer-zero-width | tampered | standard | standard | likely_tampered | 2.3 |
| smollm2-360m/tokenizer-zero-width | tampered | thorough | thorough | likely_tampered | 2.6 |
| smollm2-360m/tokenizer-zero-width | tampered | deep | deep | likely_tampered | 2.6 |
| smollm2-360m/unsloth | presumed_benign | basic | basic | clean | 1.8 |
| smollm2-360m/unsloth | presumed_benign | standard | standard | clean | 2.9 |
| smollm2-360m/unsloth | presumed_benign | thorough | thorough | clean | 2.8 |
| smollm2-360m/unsloth | presumed_benign | deep | deep | clean | 3.0 |

## Limits of this evaluation

- Built attacks of one kind share a builder, so they are correlated: an interval over five seeds of one builder is narrower than five independent attacks would earn.
- Calibration and accuracy are measured on the same corpus (in-sample). Families added later are the out-of-sample check.
- Community re-uploads are presumed benign, not proven benign.
- Trained-in behaviour is evaluated only where probe runs exist (the Qwen3-0.6B fixtures); the other families are static and provenance only.
