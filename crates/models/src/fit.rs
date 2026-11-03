//! Line fits by linear programming: least absolute deviations (Charnes, Cooper and Ferguson,
//! Management Science 1, 1955) and minimax deviation. H. P. Williams, *Model Building in
//! Mathematical Programming* (5th ed., 2013), problem "Curve fitting". Source: Gurobi
//! modeling-examples `curve_fitting`.
//!
//!   y_i = α + β x_i + u_i − v_i,   u, v ≥ 0,   α, β free;
//!   L1: min Σ_i (u_i + v_i);   L∞: min z with z ≥ u_i, z ≥ v_i.
//!
//! Columns: α, β, then u_i, v_i by observation, then z (L∞ only).

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub pts: Vec<(f64, f64)>,
}

/// The L1 model, or with `cheb` the L∞ model.
pub fn model(d: &Data, cheb: bool) -> Model {
    let c = if cheb { 0.0 } else { 1.0 };
    let mut m = Model::new(false);
    let (a, b) = (m.var(-INF, INF, 0.0), m.var(-INF, INF, 0.0));
    let uv: Vec<(usize, usize)> = d.pts.iter().map(|_| (m.var(0.0, INF, c), m.var(0.0, INF, c))).collect();
    for (&(x, y), &(u, v)) in d.pts.iter().zip(&uv) {
        m.eq([(a, 1.0), (b, x), (u, 1.0), (v, -1.0)], y);
    }
    if cheb {
        let z = m.var(0.0, INF, 1.0);
        for &(u, v) in &uv {
            m.le([(u, 1.0), (z, -1.0)], 0.0);
            m.le([(v, 1.0), (z, -1.0)], 0.0);
        }
    }
    m
}

pub fn run(d: &Data) -> Out {
    let mut text = String::new();
    let mut obj = Vec::new();
    let mut last = None;
    for cheb in [false, true] {
        let m = model(d, cheb);
        let s = m.solve();
        if !s.x.is_empty() {
            let _ = writeln!(
                text,
                "{}: y = {:.4} + {:.4} x, deviation {:.5}",
                if cheb { "L∞" } else { "L1" },
                s.x[0],
                s.x[1],
                s.obj
            );
        }
        obj.push(s.obj);
        last = Some((m, s));
    }
    let (m, s) = last.unwrap();
    Out::of(&m, &s, obj, text)
}
