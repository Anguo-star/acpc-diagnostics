# Paper 1 remediation Phase 0 snapshot

- Created UTC: `2026-07-11T02:53:20.194248+00:00`
- Local branch / HEAD: `ag/dev` / `c943fdf75cd71bc08e5466e1700676069728b7d2`
- Required baseline: `c943fdf75cd71bc08e5466e1700676069728b7d2`
- Baseline alignment: **exact**
- Target plan blob: **verified**

## Immutable inputs

- Files audited: 34
- Missing or invalid: 0

## Legacy ATR semantics

The current `acpc_h_l2_p90` is q90 over flattened `B×H` step tokens, followed downstream by checkpoint-level clean-transition-median normalization. It is not one stacked horizon radius per anchor.

## PLDM checkpoint audit

- Manifest rows: 36 / 36
- Resolved model files: 36
- Missing model files: 0
- Status: **ready_for_deserialization_smoke**
- Full CPU deserialization: **verified**

## Dataset audit

- Resolved H5 datasets: 4 / 4
- Status: **ready**

## Target-view audit

- Summary JSON: `ok`
- Canonical manifest builder exists: `True`
- Structured metadata present: `True`
- Canonical rows / pairs: 64 / 32
- Status: **verified**

## Enumerated stressor severities

- Gaussian blur: 1, 3, 7, 11, 15
- Resize: 1.0, 0.75, 0.5, 0.25
- V2 severity selection remains unfrozen; no behavior results were used.

## Gate 0

Status: **pass**


The runtime smoke and target-view manifest are accepted only when their stored schema, row counts, checks, hashes, and current builder binding all validate.
