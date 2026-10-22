# Cross-Check Report

An account of how this edition was audited and expanded, what changed, and what is still missing. Produced with a plan, write, audit loop: a lead agent audited the repository against a total LLMOps concept map; five writer agents wrote the new material from precise outlines; a fixer corrected defects in the original chapters; automated checks verified the result.

## What was checked

| Check | Method | Result |
|---|---|---|
| Structure | `scripts/audit.py`: every numbered `N.M` section has Conceptual Deep Dive, Cross-Connection Scenarios, Production Anti-Patterns, Minimal Working Code and Interview Angle, plus at least one code block; required-concept keywords per module | 0 issues |
| Python syntax | every ```python block parsed with `ast` | 0 errors |
| Runtime | stdlib/numpy-only sections executed in a clean directory | 83 sections ran, 0 failed, 15 skipped (need third-party packages) |
| Cross-references | `scripts/check_xrefs.py` validates "Module NN section X.Y" and "section N.M" references against real headings; sensitivity confirmed by planting broken references | 254 checked, 0 broken |
| PDFs | one per module, Unicode fonts embedded | 20 / 20, no missing glyphs |
| Site | static build, Mermaid diagrams rendered in the reader | 20 modules, 142 episodes |

## What changed

| Area | Change |
|---|---|
| Structure | Modules moved to `chapters/` with underscore names; README links updated; `scripts/`, `extensions/`, `pdfs/`, `site/`, `docs/` added |
| Original modules 01-12 | 32 new extension sections (`N.6` and up) in the five-part concept format |
| New modules 13-20 | CI/CD and release engineering; evaluation operations and the data flywheel; fine-tuning and alignment operations; governance, compliance and responsible AI; platform engineering, multi-tenancy and incident response; production projects and labs; learning roadmap and study plan; mock-interview and system-design question bank (50 sections) |
| Tooling | Audit, cross-reference checker, snippet runner, PDF and site builders; fence-aware section parsing so headings inside code samples are not counted |

## Defects found in the original chapters

| # | Where | Problem | Status |
|---|---|---|---|
| 1 | Module 07 section 2.5 | Sample size for a 2-point difference stated as about 1,500; the module's own formula gives about 5,000 per arm (15.7 x 0.1275 / 0.0004 = 5,004) | Fixed |
| 2 | Module 12 section 1.6 | "n=100 gives +-6 points" holds near p=0.9 but is about +-10 at p=0.5 | Fixed |
| 3 | Module 10 | OpenTelemetry GenAI attribute names mixed across spec revisions | Hedged: "verify current spec" note, `gen_ai.provider.name` leads the table |
| 4 | Module 12 section 1.6 | Prefill throughput row inconsistent with the roofline derivation (0.5 x 989e12 / 16e9 = 30.9k tok/s BF16, about 62k FP8) | Fixed |
| 5 | `docker-compose.yml`, Modules 10 and 12 | Floating image tags contradict the pinning rule | Hedged: comments added, no invented versions |
| 6 | `docker-compose.yml` | `runtime: nvidia` together with `deploy.devices` | Fixed: legacy `runtime` removed. **Untested on a GPU host** |
| 7 | Module 11 section 2.4 | A router cost/quality figure presented as fact | Fixed: replaced by "measure on your own traffic" |
| 8 | Modules 01, 05, 08, 11 | Prices and latencies stated without labels | Hedged: labelled illustrative or perishable; arithmetic re-checked |
| 9-10 | Modules 08, 09 | OWASP list and MCP spec details can change | Hedged: edition and "check current spec" notes, pointers to new sections |
| 11-13 | Numbering, scenario count, README | Needed new content or a README change | Addressed by the new sections and this README |

## Known remaining gaps

- The original-chapter audit sampled heavily and did not read every line; its issue list is not exhaustive.
- Third-party-library snippets (vLLM, LangGraph, TRL, MCP SDK, Qdrant, LangFuse, and similar) are syntax-checked only. Flags and APIs drift between releases; check current documentation.
- Prices, benchmark ranks, GPU figures, regulation details and protocol versions are perishable and deliberately generic or labelled.
- Regulatory and standards content (EU AI Act, NIST AI RMF, ISO/IEC 42001) is orientation only, not legal advice; verify current text.
- Not covered in depth: hardware-specific kernel tuning, vendor-specific managed-service configuration, and jurisdiction-specific legal detail.
- The audit is structural. It cannot prove that an explanation is correct; human technical review remains valuable.
