//! Index tracking: a long-only portfolio of sampled S&P 100 stocks whose daily returns follow the
//! index, by least squares, then with at most K holdings. Source: Gurobi modeling-examples
//! `index_tracking`.
//!
//! Days t, assets i with returns r_ti, index return m_t. Columns: weights w_i ∈ [0, ū], and in the
//! second model z_i holding i, binary:
//!
//!   Σ_i w_i = 1;   min Σ_t (Σ_i r_ti w_i − m_t)²            (ū = 1)
//!   Σ_i w_i = 1,   w_i ≤ z_i,   Σ_i z_i ≤ K;   same objective   (ū = 0.2)
//!
//! The first is a convex QP (HiGHS), the second a MIQP (SCIP). The notebook stops the second at
//! Gurobi's default 1e-4 gap (it reports 2.1e-5) and prints weights only to 0.01%, so its printed
//! weights are the check (test), not an objective.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub tickers: Vec<String>,
    /// r[t][i].
    pub returns: Vec<Vec<f64>>,
    pub index: Vec<f64>,
    pub max_assets: f64,
    pub cap: f64,
}

/// The least-squares objective through residual columns e_t = 100 (r_t·w − m_t), in percent:
/// min Σ_t e_t² / 10⁴. Expanded as w'(R'R)w − 2 (R'm)'w + m'm its terms (about 0.14) nearly
/// cancel to the optimum (about 0.01), which SCIP could not close in minutes (measured); the
/// residual form is diagonal. Bounds |e_t| ≤ 100 (|r|, |m| < 0.5 here, Σ w = 1) keep it bounded.
fn objective(m: &mut Model, d: &Data) {
    let n = d.tickers.len();
    let top = d.returns.iter().flatten().chain(&d.index).fold(0.0, |a: f64, v| a.max(v.abs()));
    assert!(top < 0.5, "a daily return of {top} breaks the residual bound");
    for (t, r) in d.returns.iter().enumerate() {
        let e = m.var(-100.0, 100.0, 0.0);
        m.eq((0..n).map(|i| (i, 100.0 * r[i])).chain([(e, -1.0)]), 100.0 * d.index[t]);
        m.qobj(e, e, 1e-4);
    }
}

/// `sparse`: the cardinality model.
pub fn model(d: &Data, sparse: bool) -> Model {
    let n = d.tickers.len();
    let mut m = Model::new(false);
    let ub = if sparse { d.cap } else { 1.0 };
    for _ in 0..n {
        m.var(0.0, ub, 0.0);
    }
    m.eq((0..n).map(|i| (i, 1.0)), 1.0);
    if sparse {
        let z: Vec<usize> = (0..n).map(|_| m.bin(0.0)).collect();
        for (i, &zi) in z.iter().enumerate() {
            m.le([(i, 1.0), (zi, -1.0)], 0.0);
        }
        m.le(z.iter().map(|&j| (j, 1.0)), d.max_assets);
    }
    objective(&mut m, d);
    m
}

pub fn run(d: &Data) -> Out {
    let mut runs = Vec::new();
    let mut text = String::new();
    for sparse in [false, true] {
        let m = model(d, sparse);
        let s = m.solve();
        let w: Vec<String> = d
            .tickers
            .iter()
            .zip(&s.x)
            .filter(|&(_, &w)| w > 1e-6)
            .map(|(t, w)| format!("{t} {:.2}%", 100.0 * w))
            .collect();
        let _ = writeln!(
            text,
            "{} squared error {:.6e}: {}",
            if sparse { "at most K" } else { "all" },
            s.obj,
            w.join(", ")
        );
        runs.push((m, s));
    }
    Out::all(&runs, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both optima print as the notebook's weights to 0.01%.
    #[test]
    #[ignore = "data/index_tracking.json is not committed until SCIP solves the MIQP in reasonable time"]
    fn weights_match_notebook() {
        let p = format!("{}/../../data/index_tracking.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let d: Data = serde_json::from_value(v["data"].clone()).unwrap();
        let printed: Vec<Vec<f64>> = serde_json::from_value(v["printed"].clone()).unwrap();
        for (sparse, want) in [false, true].into_iter().zip(&printed) {
            let s = model(&d, sparse).solve();
            for (i, &p) in want.iter().enumerate() {
                assert!(
                    (100.0 * s.x[i] - p).abs() <= 0.005 + 1e-9,
                    "{} {}: {} against {p}",
                    sparse,
                    d.tickers[i],
                    100.0 * s.x[i]
                );
            }
        }
    }
}
