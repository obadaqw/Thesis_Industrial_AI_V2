# Thesis LaTeX Project — Build Instructions

## Quick start (Overleaf — recommended for a first LaTeX project)

1. Create a new Overleaf project → "Upload Project" → upload this whole folder as a zip.
2. Menu (top left) → Compiler → set to **XeLaTeX** (not pdfLaTeX — required, see below).
3. Click Recompile. First compile may take ~30–60s (installs nothing extra; all packages used here are standard TeX Live and already on Overleaf).
4. If cross-references, the ToC, or citation numbers show as "??", just click Recompile 2–3 more times. LaTeX resolves these iteratively; this is normal, not an error.

## Quick start (local)

Requires a full TeX Live install (2023 or later) with XeLaTeX and `latexmk`.

```bash
cd thesis_latex
latexmk -xelatex main.tex
```

or manually, three passes (needed to settle the ToC, citations, and cross-refs):

```bash
xelatex main.tex
xelatex main.tex
xelatex main.tex
```

Output: `main.pdf`.

## Why XeLaTeX and not pdfLaTeX

The drafted chapters contain raw Unicode symbols (σ, µ, °, ±, ≥, ≤, →, ↑, ↓, √, Δ) copied
directly from the reviewed markdown/docx chapters. XeLaTeX renders any Unicode codepoint
the loaded font supports, with no `inputenc`/`fontenc` configuration needed. pdfLaTeX
would require converting every one of these to a LaTeX math command
(`\sigma`, `\pm`, `\geq`, ...) by hand — doable, but not worth doing for a first project.

## Project structure

```
main.tex              — top-level document, \input's everything below
preamble.tex           — every package and custom command, with inline comments
references.tex         — consolidated bibliography [1]-[91] (plain thebibliography)
frontmatter/            — title page, declaration, abstracts, abbreviations
chapters/ch1.tex ... ch6.tex
appendices/appA.tex ... appG.tex
main.pdf               — a compiled reference copy (184 pages), so you can confirm
                          your own compile matches before you start editing
```

## What was already fixed for you

The chapters were converted from markdown via `pandoc`, then hand-corrected for a
clean XeLaTeX compile. If you see any of these again after editing, here's why they
happened the first time and how they were fixed:

- **`\rq` already defined** — `\rq` is a built-in LaTeX primitive (right quote); a
  custom command of the same name in `preamble.tex` was renamed to `\rqref`.
- **`bidi` package errors on load order** — the Arabic abstract deliberately does
  *not* use `polyglossia`+`bidi` (see the long comment in `preamble.tex` and in
  `frontmatter/abstract_ar.tex`). It instead `\includepdf`'s a single pre-rendered
  page (`frontmatter/abstract_ar.pdf`), which sidesteps bidi's fragile package
  ordering rules entirely. If you need to edit the Arabic wording, edit the Word
  version and re-export that one page as a PDF, or read the comment in
  `preamble.tex` for the native-LaTeX alternative.
- **Missing-character warnings** (₀, ☐, ′, ∈, ⁻⁴, ✅, ●, box-drawing `─` in code
  listings) — these are Unicode characters the main text font (TeX Gyre Termes)
  doesn't contain glyphs for. Each was either converted to a proper LaTeX command
  (`\textsubscript{0}`, `$\square$`, `$\in$`, `\checkmark`, `$\bullet$`, etc.) or,
  for the box-drawing dividers inside Appendix E's code listings, fixed by
  switching the monospace font to DejaVu Sans Mono, which has full glyph coverage.
- **`\tightlist` undefined** — a command pandoc emits for compact markdown lists;
  defined as a no-op in `preamble.tex`.
- **Duplicate "References" sections inside Chapters 2, 3 and 5** — the original
  markdown chapters each ended with their own `## References` list (needed when
  they were delivered as separate Word documents per chapter). These were stripped
  before conversion so the bibliography appears exactly once, consolidated, as its
  own chapter after Chapter 6.

## Known placeholders still in the text

Search the project for `[Insert` and `TODO` — these mark exactly the same gaps
flagged in the chat conversation that produced this thesis: supervisor name,
submission date, CPU/RAM specs (§4.1.1), several `xai_timing.json`-sourced
values in §4.3.3, the Palestinian GDP statistic in §1.1, and the expert
evaluation results in §4.8 (currently a full placeholder section).

## Upgrading the bibliography to biblatex later (optional)

`references.tex` uses plain `thebibliography` rather than `biblatex`, so that the
numbering [1]-[91] already proofread in the drafted chapters is reproduced exactly
with no bib-parsing risk. If you add many more references and want alphabetical
sorting, automatic style switching, etc., migrate to `biblatex` + `biber`:
convert each `\bibitem{refN} ...` line into a proper `.bib` entry (type `@article`,
`@inproceedings`, `@book`, etc. as appropriate), then replace `\input{references}`
in `main.tex` with `\printbibliography` and add `\usepackage{biblatex}` +
`\addbibresource{references.bib}` to `preamble.tex`.
