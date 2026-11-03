//! Food truck siting: where to park trucks to serve buildings. Source: Gurobi modeling-examples
//! `burrito_optimization_game` (round 1, days 1–4; the notebook solves day 1).
//!
//! Truck spots t, buildings b, and pairs (b, t) with expected sales σ_bt > 0 (demand scaled by
//! distance); margin p − c per burrito, cost K per truck. Columns: x_t truck placed, y_bt building b
//! served by t, binary:
//!
//!   y_bt ≤ x_t,   Σ_t y_bt ≤ 1;   max (p − c) Σ σ_bt y_bt − K Σ x_t.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Data {
    pub price: f64,
    pub ingredient: f64,
    pub truck: f64,
    pub spots: Vec<String>,
    pub buildings: Vec<String>,
    /// (building, spot, σ).
    pub pairs: Vec<(String, String, f64)>,
}

pub fn model(d: &Data) -> Model {
    let (sp, bd) = (Names::new(d.spots.iter().map(String::as_str)), Names::new(d.buildings.iter().map(String::as_str)));
    let mut m = Model::new(true);
    let x: Vec<usize> = d.spots.iter().map(|_| m.bin(-d.truck)).collect();
    let mut by = vec![Vec::new(); d.buildings.len()];
    for (b, t, s) in &d.pairs {
        let y = m.bin((d.price - d.ingredient) * s);
        m.le([(y, 1.0), (x[sp.at(t)], -1.0)], 0.0);
        by[bd.at(b)].push((y, 1.0));
    }
    for r in by {
        m.le(r, 1.0);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    let placed: Vec<&str> = d.spots.iter().zip(&s.x).filter(|&(_, &v)| v > 0.5).map(|(t, _)| t.as_str()).collect();
    let _ = writeln!(text, "profit {:.0}, trucks {}", s.obj, placed.join(" "));
    Out::of(&m, &s, vec![s.obj], text)
}
