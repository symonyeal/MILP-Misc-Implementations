# Working contract

- A case is one module in `crates/models/src` (data struct, `model`, `run`) plus one `data/*.json`.
  Register it in `models::run` and `models::model`. Its header states the problem, its source and the
  formulation in the source's notation. Formulations come from a cited source, never from memory.
- Shared structure has one home: model building and solving (HiGHS, SCIP) in `milp`, hypergraphs in `hyper`, rows that
  more than one case uses in `models/src/lib.rs` (`stock`, `grid`, `Names`).
- `milp` and `hyper` do no I/O. Only `main.rs` and `tools/` read or write files.
- Acceptance is `Out::pass`: optimal, `Model::viol` ≤ 1e-6, and every recorded Gurobi objective interval met to
  1e-6 relative. `cargo test --release` runs it on every data file. Never weaken an oracle to pass.
- Checks before a commit: `cargo fmt --check`, `cargo clippy --release --all-targets -- -D warnings`,
  `cargo test --release`, `python -m ruff check tools`, `python -m ruff format --check tools`.
- Windows: build with `CARGO_TARGET_DIR` on a `subst` drive, because HiGHS's MSBuild tree exceeds the
  path limit.
