# Advisory Rust structure checks

Sentrux 0.5.7 scans the first-party Rust files and Cargo manifests selected by `scope.json`.
The `milp` and `hyper` adapters may not depend on case models. Its layer convention assigns
the adapters a larger order than the case layer. Feasibility and recorded oracle intervals remain acceptance.

Use the shared `Claude Func Folder/sentrux/scan.py` helper:

```text
python "<shared runner>" "<repository>" "<new run label>"
```

It copies tracked source with current working-tree bytes into an indexed scan directory,
checks rules and compares `baseline.json`. Stage new source files first. Vendor sources, data,
archives, build output and integration tests are excluded; inline Rust tests remain. Logs and
source hashes stay under the runner's `evidence/` directory. Append `--baseline` only after
reviewing the proposed replacement baseline.

The runtime is pinned to 0.5.7 with analytics disabled. Other machines need that version,
its grammars and the shared helper. Direct repository scans use a different scope.

The trial rejected a planted reverse crate-root import. Qualified cross-crate imports are
not reliably resolved by this release. The shared runner now first checks Cargo's declared
normal/build workspace dependencies against these layers, including target-specific declarations.
It fails on forbidden dependencies or unassigned crates, regardless of import spelling; dev
dependencies are excluded. Sentrux metrics and within-crate graphs remain advisory; no CI gate is added.
