//! Offshore wind farm cabling: fixed-charge network flow. Source: Gurobi modeling-examples
//! `offshore_wind_farming` (after S. Fischetti and D. Pisinger, "Optimizing wind farm cable
//! routing considering power losses", European Journal of Operational Research 270, 2018).
//!
//! Vertex v supplies s_v (turbines > 0, substations < 0); a cable on edge e = (i, j) costs F_e to
//! install and k_e per unit of flow, carrying at most U_e:
//!
//!   Σ_{e out of v} x_e − Σ_{e into v} x_e = s_v,   x_e ≤ U_e y_e,   y binary;
//!   min Σ_e (k_e x_e + F_e y_e).
//!
//! Columns: y_e at e, then x_e at |E| + e.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub supply: Vec<f64>,
    pub edges: Vec<(usize, usize, f64, f64, f64)>,
}

pub fn model(d: &Data) -> Model {
    let ne = d.edges.len();
    let mut m = Model::new(false);
    for e in &d.edges {
        m.bin(e.4);
    }
    for e in &d.edges {
        m.var(0.0, INF, e.3);
    }
    for (v, &s) in d.supply.iter().enumerate() {
        let f = d.edges.iter().enumerate().filter_map(|(k, e)| {
            if e.0 == v {
                Some((ne + k, 1.0))
            } else if e.1 == v {
                Some((ne + k, -1.0))
            } else {
                None
            }
        });
        m.eq(f, s);
    }
    for (k, e) in d.edges.iter().enumerate() {
        m.le([(ne + k, 1.0), (k, -e.2)], 0.0);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let ne = d.edges.len();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "cost {}", s.obj);
        for (k, e) in d.edges.iter().enumerate().filter(|(k, _)| s.x[*k] > 0.5) {
            let _ = writeln!(text, "  cable {} -> {}: flow {}", e.0, e.1, s.x[ne + k]);
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
