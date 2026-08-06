# Two Paper Manuscripts — Build Instructions

## Files

- `paper1.tex` — Critical review: *How Do We Know a Counterfactual Is Right?* (11 pages)
- `paper2.tex` — Framework and results: *Self-Validation Is Not Validation* (16 pages)

Each is a **single self-contained file**. No preamble.tex, no references.tex, no
class files needed — unlike the thesis, everything is in the one document.

## Building

**Overleaf:** New Project → Upload Project → upload the `.tex` file on its own.
Compiler can stay on the default pdfLaTeX; no need to switch to XeLaTeX.

**Local:**
```bash
pdflatex paper1.tex
pdflatex paper1.tex     # second pass settles cross-references
```

Both were test-compiled locally with MiKTeX/pdflatex (2 passes each) after the
fixes below: paper1 clean at 11 pages, paper2 at 16 pages with the same two
cosmetic overfull-box warnings in wide tables (Table 1's Classification column
and the ablation table) plus one harmless underfull line in the bibliography.
No undefined citations or references remain after the second pass.

## Why pdfLaTeX and not XeLaTeX this time

The thesis needed XeLaTeX because pandoc had left raw Unicode (σ, ±, ≥, →) all
through the converted chapters. These two files were written directly in LaTeX,
so every symbol is a proper command (`$\theta$`, `$\geq$`, `$\pm$`). Either
compiler works; pdfLaTeX is the safer default for journal submission systems.

## Placeholders

Anything still outstanding is wrapped in `\TODO{...}` and **renders in red** in
the PDF, so nothing can be missed on a read-through. Search the source for
`TODO`. What's left is exclusively information only the author can supply —
everything derivable from the repository's own artifacts has been filled in:

- Supervisor name and correspondence email (both files)
- CRediT author contributions, funding statement, generative-AI declaration (both)
- CPU / RAM / OS specs for §3.8 of paper2 (not recorded anywhere in the
  repository — same open item as thesis `ch4.tex` §4.1.1)
- Data Availability Statement in paper2: tag a GitHub release and cite the
  commit hash once one exists
- The DOI for paper1's own bibliography entry in paper2
  (`\bibitem{alqawasema2026a}`) — add it once paper1 is posted to
  Preprints.org, per the note below

**Resolved, not placeholders anymore:**

- The `max_depth` row of the hyperparameter provenance table (paper2 Table 1)
  now states the actual reference/deviation, sourced from
  `thesis_assets/tables/csv/tbl_ch3_hyperparameter_provenance.csv` and
  `thesis_latex/appendices/appB.tex` §B.4.2: the source paper's Table 5 splits
  `max_depth` across two rows (79 for gain ratio, 140 for information gain),
  and this work pairs the information-gain criterion with the gain-ratio row's
  depth value — immaterial in practice since realised depth is 20.
- The "Features per split" row's classification was changed from "Deviation"
  to "Equivalent at $m=13$" to match that same source table — the two
  formulae give the same value only at this dataset's feature count, not in
  general.
- Both tables paper2 asked to reproduce are inserted: the six-algorithm
  comparison (Table 2, from `tbl_ch4_algorithm_comparison.md`) and the worked
  counterfactual case in engineering units (Table 4, from
  `tbl_ch4_counterfactual_case.md`).
- The four bibliography entries flagged "confirm against `references.tex`"
  (Lundberg & Lee, Rudin, Muaz et al., McNemar) have all been checked against
  `thesis_latex/references.tex` (refs 25, 24, 47, 87) and match exactly; the
  confirm-markers are removed. Muaz et al. in particular was a bare `\TODO`
  placeholder in paper1 with no citation content — now filled in.

## Figures (paper2)

Attached. `papers_latex/figures/` now ships with the five PDFs this paper
needs (copied from `thesis_assets/figures/ch4/`), and all five
`\begin{figure}` blocks are active — nothing left to uncomment. If you move
`paper2.tex` on its own (e.g. a fresh Overleaf upload), bring the `figures/`
folder along with it; `\graphicspath{{figures/}}` in the preamble expects it
alongside the `.tex` file.

Five figures is the right number for a journal article. The thesis has fifteen
for Chapter 4; the rest — drift/PSI, tier distribution, SHAP beeswarm, LIME
local, top adjusted features, per-class vs baseline, validator by split —
belong to the thesis, not to this paper.

## Moving to the MDPI template

The MDPI class (`mdpi.cls`) ships with their author package, which you download
from the journal page after choosing a target. When you have it:

1. Open their `template.tex`.
2. Copy the body of the paper — everything from `\section{Introduction}` to the
   end of Conclusions — straight across. The section structure already matches
   what MDPI expects (Introduction, Materials and Methods, Results, Discussion,
   Conclusions for paper2; Review sections for paper1).
3. Move the abstract and keywords into their `\abstract{}` and `\keyword{}`
   commands.
4. Move the back matter into their dedicated commands
   (`\authorcontributions{}`, `\funding{}`, `\dataavailability{}`,
   `\conflictsofinterest{}`).
5. Convert `\begin{thebibliography}` entries to their `\bibitem` format — the
   structure is identical, only the surrounding environment differs.

Do this **last**, after the content is settled. Editing inside a journal class
is slower and the error messages are worse.

## One thing to sort before submitting both

Resolved. Paper 2 §5.4 no longer restates the three-level validation protocol
in full — it now summarises in two sentences and cites paper 1
(`\cite{alqawasema2026a}`) for the complete development, keeping only its own
Level-2 application and results.

That citation currently points to a bare `\TODO{}` DOI placeholder because
paper 1 hasn't been posted yet. **Post paper 1 to Preprints.org on submission
and fill in the DOI in paper2.tex's `alqawasema2026a` bibitem** — that's the
one remaining step to close this out.
