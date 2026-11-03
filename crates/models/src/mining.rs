//! Mining: H. P. Williams, *Model Building in Mathematical Programming* (5th ed., 2013), problem
//! "Mining". Source: Gurobi modeling-examples `mining`.
//!
//! Year t, mine k (royalty R_k while open, capacity C_k, ore quality q_k): extract e_tk, blend
//! b_t, mine worked w_tk ∈ {0, 1}, mine still open a_tk ∈ {0, 1}; discount δ_t = (1 + ρ)^−(t−1).
//!
//!   Σ_k w_tk ≤ N,   Σ_k q_k e_tk = Q_t b_t,   Σ_k e_tk = b_t,   e_tk ≤ C_k w_tk,   w_tk ≤ a_tk,
//!   a_t+1,k ≤ a_tk;   max Σ_t δ_t (P b_t − Σ_k R_k a_tk).

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Out, grid};

#[derive(Deserialize)]
pub struct Data {
    pub mines: Vec<Mine>,
    pub target: Vec<f64>,
    pub rate: f64,
    pub max_open: f64,
    pub price: f64,
}

#[derive(Deserialize)]
pub struct Mine {
    pub royalty: f64,
    pub cap: f64,
    pub quality: f64,
}

/// The model, and the open columns a[t][k].
pub fn model(d: &Data) -> (Model, Vec<Vec<usize>>) {
    let (nt, nk) = (d.target.len(), d.mines.len());
    let disc = |t: usize| (1.0 / (1.0 + d.rate)).powi(t as i32);
    let mut m = Model::new(true);
    let b: Vec<usize> = (0..nt).map(|t| m.var(0.0, INF, d.price * disc(t))).collect();
    let e = grid(&mut m, nt, nk, |m, _, _| m.var(0.0, INF, 0.0));
    let w = grid(&mut m, nt, nk, |m, _, _| m.bin(0.0));
    let a = grid(&mut m, nt, nk, |m, t, k| m.bin(-d.mines[k].royalty * disc(t)));
    for t in 0..nt {
        m.le((0..nk).map(|k| (w[t][k], 1.0)), d.max_open);
        m.eq((0..nk).map(|k| (e[t][k], d.mines[k].quality)).chain([(b[t], -d.target[t])]), 0.0);
        m.eq((0..nk).map(|k| (e[t][k], 1.0)).chain([(b[t], -1.0)]), 0.0);
        for (k, mn) in d.mines.iter().enumerate() {
            m.le([(e[t][k], 1.0), (w[t][k], -mn.cap)], 0.0);
            m.le([(w[t][k], 1.0), (a[t][k], -1.0)], 0.0);
            if t + 1 < nt {
                m.le([(a[t + 1][k], 1.0), (a[t][k], -1.0)], 0.0);
            }
        }
    }
    (m, a)
}

pub fn run(d: &Data) -> Out {
    let (m, a) = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "discounted profit {:.2}", s.obj);
        for (t, row) in a.iter().enumerate() {
            let open: Vec<usize> = (0..row.len()).filter(|&k| s.x[row[k]] > 0.5).map(|k| k + 1).collect();
            let _ = writeln!(text, "  year {}: mines open {open:?}", t + 1);
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
