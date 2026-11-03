//! Every case file solves, replays within 1e-6 and meets the Gurobi objectives its notebook
//! recorded (`Out::pass`).

use std::path::Path;

use milp::{Backend, Status};

#[test]
fn every_case_passes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 31);
    let fail: Vec<String> = paths
        .iter()
        .filter_map(|p| {
            let (case, data, oracle) = models::read(p).unwrap();
            let o = models::run(&case, data).unwrap();
            (!o.pass(&oracle))
                .then(|| format!("{}: {:?} vs {oracle:?}, {:?}, viol {}", p.display(), o.obj, o.st, o.viol))
        })
        .collect();
    assert!(fail.is_empty(), "{fail:#?}");
}

/// Every case's first model has the same optimal value on HiGHS and on SCIP (to 1e-6 relative):
/// two independent solvers on the same rows. Linear models only.
#[test]
fn scip_agrees_with_highs() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let mut bad = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let (case, data, _) = models::read(&p).unwrap();
        let m = models::model(&case, data).unwrap();
        if !m.linear() {
            continue;
        }
        let (h, s) = (m.solve_on(Backend::Highs), m.solve_on(Backend::Scip));
        if h.st != Status::Optimal || s.st != Status::Optimal || (h.obj - s.obj).abs() > 1e-6 * h.obj.abs().max(1.0) {
            bad.push(format!("{}: HiGHS {:?} {}, SCIP {:?} {}", p.display(), h.st, h.obj, s.st, s.obj));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
