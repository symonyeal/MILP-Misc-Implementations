//! Optimizing a constraint: the simplest equivalent form of a 0–1 knapsack row. H. P. Williams,
//! *Model Building in Mathematical Programming* (5th ed., 2013), problem "Optimizing a
//! constraint". Source: Gurobi modeling-examples `constraint_optimization`.
//!
//! Find integer-valued coefficients b_1 ≥ … ≥ b_8 and right-hand side b_0 of Σ b_i x_i ≤ b_0 with
//! the same 0–1 solutions as the original row: every minimal cover R of the original must
//! violate it (Σ_{i∈R} b_i ≥ b_0 + 1) and every maximal feasible set C satisfy it
//! (Σ_{i∈C} b_i ≤ b_0). The sets are the notebook's; so is the objective, b_0 − b_3 − b_5, which
//! reflects the variables the notebook complemented. An LP, b ≥ 0 continuous, as the notebook.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub n: usize,
    pub le: Vec<Vec<usize>>,
    pub ge: Vec<Vec<usize>>,
    pub obj: Vec<(usize, f64)>,
}

pub fn model(d: &Data) -> Model {
    let mut m = Model::new(false);
    for _ in 0..d.n {
        m.var(0.0, INF, 0.0);
    }
    for &(j, c) in &d.obj {
        m.c[j] = c;
    }
    for s in &d.le {
        m.le(s.iter().map(|&i| (i, 1.0)).chain([(0, -1.0)]), 0.0);
    }
    for s in &d.ge {
        m.ge(s.iter().map(|&i| (i, 1.0)).chain([(0, -1.0)]), 1.0);
    }
    for i in 1..d.n - 1 {
        m.ge([(i, 1.0), (i + 1, -1.0)], 0.0);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let b: Vec<String> = s.x.iter().map(|v| format!("{v}")).collect();
        let _ = writeln!(text, "objective {}; b = [{}]", s.obj, b.join(", "));
    }
    Out::of(&m, &s, vec![s.obj], text)
}
