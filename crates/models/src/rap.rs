//! Resource assignment with a budget. Source: Gurobi modeling-examples `milp_tutorial` (RAP,
//! problem 3, the recorded three-person instance).
//!
//! Assign resources r to jobs j (x_rj = 1) for matching score s_rj at cost c_rj within budget B;
//! a job left open costs M through its gap g_j:
//!
//!   Σ_r x_rj + g_j = 1,   Σ_j x_rj ≤ 1,   Σ c_rj x_rj ≤ B;   max Σ s_rj x_rj − M Σ_j g_j,
//!   x binary, g ≥ 0.
//!
//! Columns: x by pair, then g_j.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Data {
    pub resources: Vec<String>,
    pub jobs: Vec<String>,
    pub pairs: Vec<(String, String, f64, f64)>,
    pub budget: f64,
    pub gap_penalty: f64,
}

pub fn model(d: &Data) -> Model {
    let (rn, jn) = (Names::new(d.resources.iter().map(String::as_str)), Names::new(d.jobs.iter().map(String::as_str)));
    let mut m = Model::new(true);
    let x: Vec<(usize, usize, usize)> = d.pairs.iter().map(|(r, j, s, _)| (m.bin(*s), rn.at(r), jn.at(j))).collect();
    let g: Vec<usize> = d.jobs.iter().map(|_| m.var(0.0, INF, -d.gap_penalty)).collect();
    for (j, &gj) in g.iter().enumerate() {
        m.eq(x.iter().filter(|p| p.2 == j).map(|p| (p.0, 1.0)).chain([(gj, 1.0)]), 1.0);
    }
    for r in 0..d.resources.len() {
        m.le(x.iter().filter(|p| p.1 == r).map(|p| (p.0, 1.0)), 1.0);
    }
    m.le(x.iter().zip(&d.pairs).map(|(p, q)| (p.0, q.3)), d.budget);
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let on: Vec<String> = d
            .pairs
            .iter()
            .enumerate()
            .filter(|(k, _)| s.x[*k] > 0.5)
            .map(|(_, p)| format!("{} -> {}", p.0, p.1))
            .collect();
        let _ = writeln!(text, "objective {}; {}", s.obj, on.join(", "));
    }
    Out::of(&m, &s, vec![s.obj], text)
}
