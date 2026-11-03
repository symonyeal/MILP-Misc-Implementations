//! Protein comparison by contact map overlap: H. P. Williams, *Model Building in Mathematical
//! Programming* (5th ed., 2013), problem "Protein comparison" (after Lancia, Carr, Walenz and
//! Istrail, RECOMB 2001). Source: Gurobi modeling-examples `protein_comparison`.
//!
//! Map nodes of graph 1 to nodes of graph 2 (x_ij = 1), each node at most once, without crossings
//! (i < k with l < j cannot both map), maximising the edges (i, k) of graph 1 whose images (j, l)
//! are edges of graph 2:
//!
//!   Σ_i x_ij ≤ 1,   Σ_j x_ij ≤ 1,   x_ij + x_kl ≤ 1 (i < k, l < j);
//!   max Σ_{(i,k) ∈ E1, (j,l) ∈ E2} x_ij x_kl,   x binary.
//!
//! The notebook keeps the products; SCIP solves the binary quadratic program.
//! Columns: x_ij at n2 (i − 1) + j − 1.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub n1: usize,
    pub e1: Vec<(usize, usize)>,
    pub n2: usize,
    pub e2: Vec<(usize, usize)>,
}

pub fn model(d: &Data) -> Model {
    let x = |i: usize, j: usize| d.n2 * (i - 1) + j - 1;
    let mut m = Model::new(true);
    for _ in 0..d.n1 * d.n2 {
        m.bin(0.0);
    }
    for j in 1..=d.n2 {
        m.le((1..=d.n1).map(|i| (x(i, j), 1.0)), 1.0);
    }
    for i in 1..=d.n1 {
        m.le((1..=d.n2).map(|j| (x(i, j), 1.0)), 1.0);
    }
    for i in 1..=d.n1 {
        for k in i + 1..=d.n1 {
            for j in 1..=d.n2 {
                for l in 1..j {
                    m.le([(x(i, j), 1.0), (x(k, l), 1.0)], 1.0);
                }
            }
        }
    }
    for &(i, k) in &d.e1 {
        for &(j, l) in &d.e2 {
            m.qobj(x(i, j), x(k, l), 1.0);
        }
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let map: Vec<String> = (1..=d.n1)
            .flat_map(|i| (1..=d.n2).map(move |j| (i, j)))
            .filter(|&(i, j)| s.x[d.n2 * (i - 1) + j - 1] > 0.5)
            .map(|(i, j)| format!("{i}->{j}"))
            .collect();
        let _ = writeln!(text, "{} shared contacts; map {}", s.obj, map.join(" "));
    }
    Out::of(&m, &s, vec![s.obj], text)
}
