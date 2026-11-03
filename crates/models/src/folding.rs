//! Protein folding on a square lattice with one fold, the hydrophobic contact maximisation of
//! H. P. Williams, *Model Building in Mathematical Programming* (5th ed., 2013), problem "Protein
//! folding" (after Forrester and Greenberg). Source: Gurobi modeling-examples `protein_folding`.
//!
//! Acids 1..n in a chain; hydrophobic acids H. A pair i < j of H with j > i + 1 can touch after
//! one fold only at k = (i + j − 1)/2 (when integral), and no fold may lie between them elsewhere:
//!
//!   f_k + m_ij ≤ 1 (i ≤ k < j, k ≠ (i + j − 1)/2),   m_ij ≤ f_(i+j−1)/2,
//!   max Σ m_ij over pairs that have a middle fold;   m, f binary.
//!
//! Columns: f_k at k − 1, then m_ij in pair order.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub n: usize,
    pub phobic: Vec<usize>,
}

/// The model and its pairs (i, j).
pub fn model(d: &Data) -> (Model, Vec<(usize, usize)>) {
    let pairs: Vec<(usize, usize)> =
        d.phobic.iter().flat_map(|&i| d.phobic.iter().filter(move |&&j| j > i + 1).map(move |&j| (i, j))).collect();
    let mut m = Model::new(true);
    for _ in 0..d.n {
        m.bin(0.0);
    }
    for &(i, j) in &pairs {
        let mid = (i + j) % 2 == 1;
        let x = m.bin(if mid { 1.0 } else { 0.0 });
        for k in i..j {
            if mid && 2 * k == i + j - 1 {
                m.le([(x, 1.0), (k - 1, -1.0)], 0.0);
            } else {
                m.le([(k - 1, 1.0), (x, 1.0)], 1.0);
            }
        }
    }
    (m, pairs)
}

pub fn run(d: &Data) -> Out {
    let (m, pairs) = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let folds: Vec<usize> = (0..d.n).filter(|&k| s.x[k] > 0.5).map(|k| k + 1).collect();
        let hits: Vec<String> = pairs
            .iter()
            .enumerate()
            .filter(|(p, _)| s.x[d.n + p] > 0.5)
            .map(|(_, (i, j))| format!("{i}-{j}"))
            .collect();
        let _ = writeln!(text, "{} contacts {}; folds after acids {folds:?}", s.obj, hits.join(" "));
    }
    Out::of(&m, &s, vec![s.obj], text)
}
