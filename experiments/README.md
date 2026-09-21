# Experiments

The measurements behind [the blog post](../blog/index.html), in the order
its argument makes them. Each folder is self-contained: the script(s) that
produced its numbers, a Kubernetes Job manifest to rerun them, and the
exact result data the blog charts render.

| Folder | Blog section | Question it answers |
|---|---|---|
| [`01-first-measurement`](./01-first-measurement/) | §2 | Does the tuned config actually help — at the kernel level, and end-to-end? |
| [`02-nsys-shape-profiling`](./02-nsys-shape-profiling/) | §3, §4 | Which shape/batch-size is responsible for the regression, and what's different about how it's measured? |
| [`03-group-size-m-patch`](./03-group-size-m-patch/) | §5 | Does fixing one parameter, by hand, at the one anchor that regressed, recover it? |
| [`04-l2-flush-retune`](./04-l2-flush-retune/) | §6 | Does the original config regress under a cold cache too — and can the tuning script itself be fixed, not just one config? |
| [`05-l2-flush-kernel-validation`](./05-l2-flush-kernel-validation/) | §6 | Is the retuned config reliably faster at the kernel level, across every shape and batch size? |
| [`06-l2-flush-e2e-validation`](./06-l2-flush-e2e-validation/) | §6 | Does that hold up in real, end-to-end serving, at every concurrency? |

This is a curated subset — the full research history (every hypothesis
tried, including the ones that didn't pan out) lives outside this repo.
These six are the ones whose results the blog actually shows.

`tuned-configs/` (repo root) is the current output of `04`'s retune —
the config this whole investigation converges on.
