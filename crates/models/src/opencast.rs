//! Opencast mining: H. P. Williams, *Model Building in Mathematical Programming* (5th ed., 2013),
//! problem "Opencast mining". Source: Gurobi modeling-examples `opencast_mining`.
//!
//! Block b, worth p_b when extracted, may be dug only after every block a above it, (b, a) ∈ A:
//! the maximum-weight closure of the precedence digraph (Picard, Management Science 22, 1976).
//!
//!   max Σ_b p_b x_b   s.t.  x_b ≤ x_a  ((b, a) ∈ A),   0 ≤ x ≤ 1.
//!
//! Each row has one +1 and one −1, so the matrix is totally unimodular and the simplex vertex is
//! integral; x stays continuous, as in the notebook.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Data {
    pub blocks: Vec<Block>,
    pub arcs: Vec<(String, String)>,
}

#[derive(Deserialize)]
pub struct Block {
    pub name: String,
    pub p: f64,
}

pub fn model(d: &Data) -> Model {
    let names = Names::new(d.blocks.iter().map(|b| b.name.as_str()));
    let mut m = Model::new(true);
    for b in &d.blocks {
        m.var(0.0, 1.0, b.p);
    }
    for (b, a) in &d.arcs {
        m.le([(names.at(b), 1.0), (names.at(a), -1.0)], 0.0);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let dug: Vec<&str> =
            d.blocks.iter().zip(&s.x).filter(|(_, x)| **x > 0.5).map(|(b, _)| b.name.as_str()).collect();
        let _ = writeln!(text, "profit {}; extract blocks {}", s.obj, dug.join(" "));
    }
    Out::of(&m, &s, vec![s.obj], text)
}
