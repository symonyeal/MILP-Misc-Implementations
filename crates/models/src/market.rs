//! Market sharing by goal programming: H. P. Williams, *Model Building in Mathematical
//! Programming* (5th ed., 2013), problem "Market sharing". Source: Gurobi modeling-examples
//! `market_sharing`.
//!
//! Allocate retailers r (δ_r = 1: to division D1) so that each goal k, Σ_r w_kr δ_r, lands on its
//! target T_k (40% of the total) with deviations u_k, v_k ≤ ε_k (5%):
//!
//!   Σ_r w_kr δ_r + u_k − v_k = T_k,   0 ≤ u_k, v_k ≤ ε_k,   δ binary;   min Σ_k (u_k + v_k).
//!
//! Columns: δ_r at r − 1 (retailers numbered from 1), then u_k, v_k by goal.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub retailers: usize,
    pub goals: Vec<Goal>,
}

#[derive(Deserialize)]
pub struct Goal {
    pub name: String,
    pub w: Vec<(usize, f64)>,
    pub target: f64,
    pub tol: f64,
}

pub fn model(d: &Data) -> Model {
    let mut m = Model::new(false);
    for _ in 0..d.retailers {
        m.bin(0.0);
    }
    for g in &d.goals {
        let (u, v) = (m.var(0.0, g.tol, 1.0), m.var(0.0, g.tol, 1.0));
        m.eq(g.w.iter().map(|&(r, w)| (r - 1, w)).chain([(u, 1.0), (v, -1.0)]), g.target);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let d1: Vec<usize> = (0..d.retailers).filter(|&r| s.x[r] > 0.5).map(|r| r + 1).collect();
        let _ = writeln!(text, "total deviation {:.4}; division D1 gets retailers {d1:?}", s.obj);
        for g in &d.goals {
            let got: f64 = g.w.iter().filter(|p| s.x[p.0 - 1] > 0.5).map(|p| p.1).sum();
            let _ = writeln!(text, "  {:16} {got:8.2} (target {})", g.name, g.target);
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
