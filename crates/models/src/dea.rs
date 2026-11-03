//! Efficiency analysis by data envelopment analysis, the CCR multiplier form (A. Charnes,
//! W. W. Cooper, E. Rhodes, "Measuring the efficiency of decision making units", European
//! Journal of Operational Research 2, 1978). H. P. Williams, *Model Building in Mathematical
//! Programming* (5th ed., 2013), problem "Efficiency analysis". Source: Gurobi modeling-examples
//! `efficiency_analysis`.
//!
//! Units h with inputs x_h and outputs y_h. The efficiency of unit o is
//!
//!   max u'y_o   s.t.  u'y_h − v'x_h ≤ 0 (every h),   v'x_o = 1,   u, v ≥ 0,
//!
//! one LP per unit; 1 means efficient. Columns: u (outputs), then v (inputs).

use std::fmt::Write;

use milp::{INF, Model, Sol};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub units: Vec<Unit>,
}

#[derive(Deserialize)]
pub struct Unit {
    pub name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// The efficiency LP of unit o.
pub fn model(d: &Data, o: usize) -> Model {
    let (ni, no) = (d.inputs.len(), d.outputs.len());
    let mut m = Model::new(true);
    for r in 0..no {
        m.var(0.0, INF, d.units[o].y[r]);
    }
    for _ in 0..ni {
        m.var(0.0, INF, 0.0);
    }
    for h in &d.units {
        m.le(
            h.y.iter().enumerate().map(|(r, &y)| (r, y)).chain(h.x.iter().enumerate().map(|(i, &x)| (no + i, -x))),
            0.0,
        );
    }
    m.eq(d.units[o].x.iter().enumerate().map(|(i, &x)| (no + i, x)), 1.0);
    m
}

pub fn run(d: &Data) -> Out {
    let runs: Vec<(Model, Sol)> = (0..d.units.len())
        .map(|o| {
            let m = model(d, o);
            let s = m.solve();
            (m, s)
        })
        .collect();
    let mut text = String::new();
    let mut by: Vec<(f64, &str)> = runs.iter().zip(&d.units).map(|((_, s), u)| (s.obj, u.name.as_str())).collect();
    by.sort_by(|a, b| b.0.total_cmp(&a.0));
    let eff: Vec<&str> = by.iter().filter(|p| p.0 >= 1.0 - 1e-7).map(|p| p.1).collect();
    let _ = writeln!(text, "efficient ({}): {}", eff.len(), eff.join(" "));
    for (e, n) in by.iter().filter(|p| p.0 < 1.0 - 1e-7) {
        let _ = writeln!(text, "  {n:12} {e:.4}");
    }
    Out::all(&runs, text)
}
