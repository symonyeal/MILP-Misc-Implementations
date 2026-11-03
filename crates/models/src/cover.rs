//! Cell tower coverage: budgeted maximum coverage on a hypergraph (Church and ReVelle 1974, the
//! maximal covering location problem; Khuller, Moss and Naor 1999, its budgeted form). Source:
//! Gurobi modeling-examples `cell_tower_coverage`.
//!
//! Regions are the vertices v with population p_v; each tower site e is the hyperedge of regions it
//! covers, built at cost c_e within budget B.
//!
//!   max Σ_v p_v z_v   s.t.  z_v ≤ Σ_{e ∋ v} x_e  (v ∈ V),   Σ_e c_e x_e ≤ B,   x, z ∈ {0, 1}.
//!
//! Columns: x_e at e, then z_v at |E| + v.

use std::fmt::Write;

use hyper::Hg;
use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub p: Vec<f64>,
    pub towers: Vec<Tower>,
    pub budget: f64,
}

#[derive(Deserialize)]
pub struct Tower {
    pub e: Vec<usize>,
    pub c: f64,
}

pub fn model(d: &Data) -> Model {
    let g = Hg::new(d.p.len(), d.towers.iter().map(|t| t.e.clone()));
    let ne = g.e.len();
    let mut m = Model::new(true);
    for _ in 0..ne {
        m.bin(0.0);
    }
    for &p in &d.p {
        m.bin(p);
    }
    for (v, es) in g.nbr_e().iter().enumerate() {
        m.le([(ne + v, 1.0)].into_iter().chain(es.iter().map(|&e| (e, -1.0))), 0.0);
    }
    m.le(d.towers.iter().enumerate().map(|(e, t)| (e, t.c)), d.budget);
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let built: Vec<usize> = (0..d.towers.len()).filter(|&e| s.x[e] > 0.5).collect();
        let cost: f64 = built.iter().map(|&e| d.towers[e].c).sum();
        let total: f64 = d.p.iter().sum();
        let _ = writeln!(text, "towers built {built:?}, cost {cost} of budget {}", d.budget);
        let _ = writeln!(text, "population covered {} of {total} ({:.2}%)", s.obj, 100.0 * s.obj / total);
    }
    Out::of(&m, &s, vec![s.obj], text)
}
