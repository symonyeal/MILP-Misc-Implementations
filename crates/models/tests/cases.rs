//! Every case file solves, replays within 1e-6 and meets the Gurobi objectives its notebook
//! recorded (`Out::pass`).

use std::path::Path;

#[test]
fn every_case_passes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 15);
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
