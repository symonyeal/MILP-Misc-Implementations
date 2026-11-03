//! Mean–variance portfolio selection (H. Markowitz, "Portfolio selection", Journal of Finance 7,
//! 1952). Source: Gurobi modeling-examples `portfolio_selection_optimization`.
//!
//! Weights x ≥ 0 over stocks with mean daily return μ and covariance Σ:
//!
//!   minimum risk:  min x'Σx  s.t.  Σ_i x_i = 1;
//!   frontier:      the same with μ'x = r, for r at `frontier` evenly spaced points from min μ to
//!                  max μ (both included), as the notebook's np.linspace.
//!
//! A convex continuous QP, solved by HiGHS's QP solver (`Backend::Highs`): the risks are about
//! 1e-5, below what SCIP's epigraph row resolves.

use std::fmt::Write;

use milp::{Backend, INF, Model, Sol};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub stocks: Vec<String>,
    pub mu: Vec<f64>,
    pub sigma: Vec<Vec<f64>>,
    pub frontier: usize,
}

/// The minimum-risk model, with the return row μ'x = r when r is given.
pub fn model(d: &Data, r: Option<f64>) -> Model {
    let n = d.stocks.len();
    let mut m = Model::new(false);
    for _ in 0..n {
        m.var(0.0, INF, 0.0);
    }
    for j in 0..n {
        for k in 0..n {
            m.qobj(j, k, d.sigma[j][k]);
        }
    }
    m.eq((0..n).map(|j| (j, 1.0)), 1.0);
    if let Some(r) = r {
        m.eq(d.mu.iter().enumerate().map(|(j, &u)| (j, u)), r);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let (lo, hi) = d.mu.iter().fold((INF, -INF), |(l, h), &u| (l.min(u), h.max(u)));
    let k = d.frontier;
    let rs = std::iter::once(None).chain((0..k).map(|t| Some(lo + (hi - lo) * t as f64 / (k - 1) as f64)));
    let runs: Vec<(Model, Sol)> = rs
        .map(|r| {
            let m = model(d, r);
            let s = m.solve_on(Backend::Highs);
            (m, s)
        })
        .collect();
    let mut text = String::new();
    let s = &runs[0].1;
    if !s.x.is_empty() {
        let ret: f64 = d.mu.iter().zip(&s.x).map(|(u, x)| u * x).sum();
        let _ = writeln!(text, "minimum risk: volatility {:.6}, expected daily return {:.6}", s.obj.sqrt(), ret);
        for (n, x) in d.stocks.iter().zip(&s.x) {
            if *x > 1e-4 {
                let _ = writeln!(text, "  {n:6} {x:.4}");
            }
        }
        let _ = writeln!(text, "frontier: {} points from return {lo:.6} to {hi:.6}", k);
    }
    Out::all(&runs, text)
}
