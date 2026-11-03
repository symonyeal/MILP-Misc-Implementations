//! Economic planning, a dynamic Leontief input–output model: H. P. Williams, *Model Building in
//! Mathematical Programming* (5th ed., 2013), problem "Economic planning". Source: Gurobi
//! modeling-examples `economic_planning`.
//!
//! Industries i; a_ij input from i per unit output of j, b_ij input from i per unit of new
//! capacity of j; labour ℓ^a_j per unit output and ℓ^b_j per unit capacity; demand d, initial
//! stocks s⁰ and capacity k⁰.
//!
//! Static model: x = A x + d (feasibility; objective 0).
//! Dynamic model, years t = 1..6, output x_it ≥ 0 (x_i1 = 0), stock s_it (t ≤ 5), new capacity
//! e_it (zero outside t = 3..5):
//!   s⁰_i = Σ_j a_ij x_j2 + Σ_j b_ij e_j3 + d_i + s_i1,
//!   x_it + s_i,t−1 = Σ_j a_ij x_j,t+1 + Σ_j b_ij e_j,t+2 + d_i + s_it   (t = 2..4),
//!   x_i5 + s_i4 = Σ_j a_ij x_j6 + d_i + s_i5,   x_i6 ≥ x̄_i (the static output),
//!   x_it − Σ_{τ≤t} e_iτ ≤ k⁰_i   (t = 1..5);
//!   max Σ_{t≤5} Σ_j (ℓ^a_j x_jt + ℓ^b_j e_jt).
//! The notebook also passes `ub=capacity0`, a dict keyed by industry for columns keyed by
//! (industry, year); gurobipy applied no bound from it (its log: bounds range [0, 0]; its plan makes
//! 961 coal in year 5 against k⁰ = 300), so capacity binds only through the rows above.

use std::fmt::Write;

use milp::{INF, Model, Sol};
use serde::Deserialize;

use crate::{Out, grid};

#[derive(Deserialize)]
pub struct Data {
    pub industries: Vec<String>,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub la: Vec<f64>,
    pub lb: Vec<f64>,
    pub s0: Vec<f64>,
    pub k0: Vec<f64>,
    pub d: Vec<f64>,
}

/// The static model x − A x = d.
pub fn model(d: &Data) -> Model {
    let n = d.industries.len();
    let mut m = Model::new(false);
    for _ in 0..n {
        m.var(0.0, INF, 0.0);
    }
    for i in 0..n {
        m.eq((0..n).map(|j| (j, if i == j { 1.0 } else { 0.0 } - d.a[i][j])), d.d[i]);
    }
    m
}

/// The dynamic model given the static output xs; columns x[i][t], s[i][t], e[i][t], t 0-based.
pub fn dynamic(d: &Data, xs: &[f64]) -> Model {
    let n = d.industries.len();
    let mut m = Model::new(true);
    let x = grid(&mut m, n, 6, |m, i, t| m.var(0.0, if t == 0 { 0.0 } else { INF }, if t < 5 { d.la[i] } else { 0.0 }));
    let s = grid(&mut m, n, 5, |m, _, _| m.var(0.0, INF, 0.0));
    let e = grid(&mut m, n, 6, |m, i, t| {
        m.var(0.0, if (2..5).contains(&t) { INF } else { 0.0 }, if t < 5 { d.lb[i] } else { 0.0 })
    });
    for i in 0..n {
        for t in 0..5 {
            // Year t + 1 balance: supply (output and stock carried in) meets next year's inputs,
            // capacity built two years ahead, demand and stock carried out.
            let mut r: Vec<(usize, f64)> = (0..n).map(|j| (x[j][t + 1], d.a[i][j])).collect();
            if t + 2 < 6 {
                r.extend((0..n).map(|j| (e[j][t + 2], d.b[i][j])));
            }
            r.push((s[i][t], 1.0));
            let rhs = if t == 0 {
                d.s0[i] - d.d[i]
            } else {
                r.extend([(x[i][t], -1.0), (s[i][t - 1], -1.0)]);
                -d.d[i]
            };
            m.eq(r, rhs);
        }
        m.ge([(x[i][5], 1.0)], xs[i]);
        for t in 0..5 {
            m.le([(x[i][t], 1.0)].into_iter().chain((0..=t).map(|u| (e[i][u], -1.0))), d.k0[i]);
        }
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m0 = model(d);
    let s0 = m0.solve();
    let mut text = String::new();
    let mut runs: Vec<(Model, Sol)> = Vec::new();
    if !s0.x.is_empty() {
        let _ = writeln!(text, "static output {:?}", s0.x.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>());
        let m = dynamic(d, &s0.x);
        let s = m.solve();
        let _ = writeln!(text, "dynamic plan: labour {:.4}", s.obj);
        runs.push((m0, s0));
        runs.push((m, s));
    } else {
        runs.push((m0, s0));
    }
    Out::all(&runs, text)
}
