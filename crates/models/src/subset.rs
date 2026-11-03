//! Best subset selection (L0-constrained least squares) as a mixed-integer QP: D. Bertsimas,
//! A. King, R. Mazumder, "Best subset selection via a modern optimization lens", Annals of
//! Statistics 44 (2016). Source: Gurobi modeling-examples `linear_regression` (l0_regression).
//!
//! Features are standardised to mean 0, variance 1, which leaves every least-squares fit's
//! residuals unchanged. With design X̃ = [x̃ 1] (intercept last), G = X̃'X̃, b = X̃'y:
//!
//!   min β'Gβ − 2b'β + y'y   s.t.  −M z_j ≤ β_j ≤ M z_j,   Σ_j z_j ≤ k,   z ∈ {0, 1},
//!
//! for k = 1..p; the objective is the residual sum of squares. The notebook bounds ‖β‖₀ with
//! Gurobi's norm constraint; these big-M rows (Bertsimas et al. §2.3) need a valid M: here
//! M = 10 max_j |β̂_j| over the full least-squares fit, and the run reports whether any |β_j|
//! reached M (then M was binding and the value is not certified). The intercept lies in
//! [min y, max y]: with centred features it is the mean of y.
//! Columns: β_j at j (0..p), the intercept at p, z_j at p + 1 + j.

use std::fmt::Write;

use milp::{Model, Sol};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub features: Vec<String>,
    pub response: String,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

/// (G, b, y'y) of the standardised design with the intercept last.
fn gram(d: &Data) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
    let (n, p) = (d.y.len(), d.features.len());
    let mut z = vec![vec![1.0; p + 1]; n];
    for j in 0..p {
        let mu = d.x.iter().map(|r| r[j]).sum::<f64>() / n as f64;
        let sd = (d.x.iter().map(|r| (r[j] - mu).powi(2)).sum::<f64>() / n as f64).sqrt();
        for i in 0..n {
            z[i][j] = (d.x[i][j] - mu) / sd;
        }
    }
    let g = (0..=p).map(|j| (0..=p).map(|k| (0..n).map(|i| z[i][j] * z[i][k]).sum()).collect()).collect();
    let b = (0..=p).map(|j| (0..n).map(|i| z[i][j] * d.y[i]).sum()).collect();
    (g, b, d.y.iter().map(|v| v * v).sum())
}

/// The least-squares fit G β = b by Cholesky factorisation (G is positive definite here).
fn ols(g: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = g[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            l[i][j] = if i == j { s.sqrt() } else { s / l[j][j] };
        }
    }
    let mut w = vec![0.0; n];
    for i in 0..n {
        w[i] = (b[i] - (0..i).map(|k| l[i][k] * w[k]).sum::<f64>()) / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (w[i] - (i + 1..n).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
    }
    x
}

/// The model for at most k features, and its M.
pub fn model(d: &Data, k: usize) -> (Model, f64) {
    let p = d.features.len();
    let (g, b, yy) = gram(d);
    let big = 10.0 * ols(&g, &b)[..p].iter().fold(0.0, |a: f64, v| a.max(v.abs()));
    let (lo, hi) = d.y.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    let mut m = Model::new(false);
    m.c0 = yy;
    for j in 0..=p {
        if j < p {
            m.var(-big, big, -2.0 * b[j])
        } else {
            m.var(lo, hi, -2.0 * b[j])
        };
    }
    for j in 0..=p {
        for l in 0..=p {
            m.qobj(j, l, g[j][l]);
        }
    }
    for j in 0..p {
        let z = m.bin(0.0);
        m.le([(j, 1.0), (z, -big)], 0.0);
        m.le([(j, -1.0), (z, -big)], 0.0);
    }
    m.le((0..p).map(|j| (p + 1 + j, 1.0)), k as f64);
    (m, big)
}

pub fn run(d: &Data) -> Out {
    let p = d.features.len();
    let mut text = String::new();
    let runs: Vec<(Model, Sol)> = (1..=p)
        .map(|k| {
            let (m, big) = model(d, k);
            let s = m.solve();
            if !s.x.is_empty() {
                let on: Vec<&str> = (0..p).filter(|&j| s.x[p + 1 + j] > 0.5).map(|j| d.features[j].as_str()).collect();
                let tight = (0..p).any(|j| s.x[j].abs() >= 0.99 * big);
                let _ = writeln!(
                    text,
                    "k = {k:2}: RSS {:.4}{}  {}",
                    s.obj,
                    if tight { " (M binding)" } else { "" },
                    on.join(" ")
                );
            }
            (m, s)
        })
        .collect();
    Out::all(&runs, text)
}
