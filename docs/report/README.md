# Project report

Publication-ready write-up of the project, for LinkedIn and Substack.

- **`substack-article.md`** — the long-form, accessible-technical article with the
  four figures embedded.
- **`linkedin-post.md`** — a short teaser post that links back to the article, with
  the suggested images to attach.
- **`figures/`** — PNGs, regenerated with `python scripts/make_figures.py` from repo root.

All numbers in the article come directly from that script (seed `20260702`), so the
prose and the figures stay in sync. Re-running the script reproduces every chart.
