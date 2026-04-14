# Audit Comparison

**AUDIT_PLAN.md wins overall.** 58 items vs 24, finer severity tiers (P0-P3), better line references, and a phased remediation plan.

## PLAN strengths
- 3x coverage — catches ADF index bug, PP formula error, decomposition normalization, ACF nil return, CSV desync, and many more that FINDINGS misses
- Separates "wrong results" (P0) from "safety issues" (P1)
- Identifies test gaps and code duplication

## FINDINGS strengths
- **H1 (MA gradient bias)** — critical bug PLAN misses entirely. The CSS optimizer treats past residuals as constants, making all MA gradients biased.
- **M4 (KPSS p-value > 1.0)** — quick fix not in PLAN
- Clearer fix guidance with "Expected behavior" descriptions

## Recommendation
Use AUDIT_PLAN.md as primary. Merge H1 and M4 from AUDIT_FINDINGS.md into it.
