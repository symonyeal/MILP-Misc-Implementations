//! Workforce scheduling with hierarchical objectives. Source: Gurobi modeling-examples `workforce`
//! (Gurobi's `workforce5` example).
//!
//! Workers w work shifts t they are available for, (w, t) ∈ A; slack s_t counts the workers shift t
//! still lacks of its need r_t. First minimise total slack; then, losing at most `reltol` of that
//! optimum, minimise the spread M − μ of shifts worked, M ≥ n_w ≥ μ for every worker.
//!
//!   Σ_{w:(w,t)∈A} x_wt + s_t = r_t,   n_w = Σ_{t:(w,t)∈A} x_wt,   x binary, s, M, μ ≥ 0;
//!   objectives (Σ_t s_t, reltol), then (M − μ).
//!
//! The notebook's max/min general constraints are the rows M ≥ n_w ≥ μ: the second objective presses
//! M down and μ up, so at its optimum they equal the max and min. n_w is substituted out.
//! Columns: x in the order of A, then s_t, then M, μ.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Data {
    pub shifts: Vec<Shift>,
    pub workers: Vec<String>,
    pub avail: Vec<(String, String)>,
    pub reltol: f64,
}

#[derive(Deserialize)]
pub struct Shift {
    pub name: String,
    pub need: f64,
}

/// The model, and the two objective vectors.
pub fn model(d: &Data) -> (Model, [Vec<f64>; 2]) {
    let (tn, wn) =
        (Names::new(d.shifts.iter().map(|s| s.name.as_str())), Names::new(d.workers.iter().map(String::as_str)));
    let a: Vec<(usize, usize)> = d.avail.iter().map(|(w, t)| (wn.at(w), tn.at(t))).collect();
    let mut m = Model::new(false);
    for _ in &a {
        m.bin(0.0);
    }
    let s0 = m.n();
    for _ in &d.shifts {
        m.var(0.0, INF, 0.0);
    }
    let (hi, lo) = (m.var(0.0, INF, 0.0), m.var(0.0, INF, 0.0));
    for (t, sh) in d.shifts.iter().enumerate() {
        let x = a.iter().enumerate().filter(|(_, p)| p.1 == t).map(|(k, _)| (k, 1.0));
        m.eq(x.chain([(s0 + t, 1.0)]), sh.need);
    }
    for w in 0..d.workers.len() {
        let x = || a.iter().enumerate().filter(move |(_, p)| p.0 == w).map(|(k, _)| (k, 1.0));
        m.le(x().chain([(hi, -1.0)]), 0.0);
        m.ge(x().chain([(lo, -1.0)]), 0.0);
    }
    let mut c1 = vec![0.0; m.n()];
    c1[s0..s0 + d.shifts.len()].fill(1.0);
    let mut c2 = vec![0.0; m.n()];
    (c2[hi], c2[lo]) = (1.0, -1.0);
    (m, [c1, c2])
}

pub fn run(d: &Data) -> Out {
    let (m, [c1, c2]) = model(d);
    let st = m.lex(&[(&c1, d.reltol), (&c2, 0.0)]);
    let s = st.last().unwrap();
    let mut text = String::new();
    if st.len() == 2 && !s.x.is_empty() {
        let slack: f64 = c1.iter().zip(&s.x).map(|(c, x)| c * x).sum();
        let _ = writeln!(text, "least total slack {}; with slack {slack} the spread of shifts is {}", st[0].obj, s.obj);
        for name in &d.workers {
            let on: Vec<&str> = d
                .avail
                .iter()
                .enumerate()
                .filter(|(k, p)| p.0 == *name && s.x[*k] > 0.5)
                .map(|(_, p)| p.1.as_str())
                .collect();
            let _ = writeln!(text, "  {name:6} {:2} shifts {}", on.len(), on.join(" "));
        }
    }
    Out::of(&m, s, st.iter().map(|s| s.obj).collect(), text)
}
