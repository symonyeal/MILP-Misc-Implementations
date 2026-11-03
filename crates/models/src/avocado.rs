//! Avocado pricing and supply allocation. Source: Gurobi modeling-examples `price_optimization`
//! ("How Much Is Too Much? Avocado Pricing and Supply Using Mathematical Optimization").
//!
//! Region r gets x_r of the supply B, priced at p_r; it sells s_r ≤ min(x_r, d_r(p_r)) and wastes
//! w_r = x_r − s_r. Demand is the notebook's linear regression,
//!   d_r(p) = α + β p + γ_r + δ (year − 2015) + η peak.
//!
//!   max Σ_r (p_r s_r − c_w w_r − c_r x_r)   s.t.  Σ_r x_r = B,   s_r ≤ x_r,   s_r − β p_r ≤ d_r(0),
//!       w_r = x_r − s_r,   0 ≤ p_r ≤ p̄,   ℓ_r ≤ x_r ≤ h_r,   s, w ≥ 0.
//!
//! The revenue p_r s_r is bilinear and nonconvex; SCIP solves it globally.
//! Columns: by region r, p at 4r, x at 4r + 1, s at 4r + 2, w at 4r + 3.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub supply: f64,
    pub waste: f64,
    pub pmax: f64,
    pub year: f64,
    pub peak: f64,
    pub fit: Fit,
    pub regions: Vec<Region>,
}

#[derive(Deserialize)]
pub struct Fit {
    pub intercept: f64,
    pub price: f64,
    pub year: f64,
    pub peak: f64,
}

#[derive(Deserialize)]
pub struct Region {
    pub name: String,
    pub shift: f64,
    pub transport: f64,
    pub lo: f64,
    pub hi: f64,
}

pub fn model(d: &Data) -> Model {
    let f = &d.fit;
    let mut m = Model::new(true);
    for g in &d.regions {
        let p = m.var(0.0, d.pmax, 0.0);
        let x = m.var(g.lo, g.hi, -g.transport);
        let s = m.var(0.0, INF, 0.0);
        let w = m.var(0.0, INF, -d.waste);
        m.qobj(p, s, 1.0);
        m.le([(s, 1.0), (x, -1.0)], 0.0);
        m.le([(s, 1.0), (p, -f.price)], f.intercept + g.shift + f.year * (d.year - 2015.0) + f.peak * d.peak);
        m.eq([(w, 1.0), (x, -1.0), (s, 1.0)], 0.0);
    }
    m.eq((0..d.regions.len()).map(|r| (4 * r + 1, 1.0)), d.supply);
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "net revenue {:.6} (millions of dollars)", s.obj);
        for (r, g) in d.regions.iter().enumerate() {
            let v = &s.x[4 * r..4 * r + 4];
            let _ = writeln!(
                text,
                "  {:21} price {:.3}, allocated {:.3}, sold {:.3}, wasted {:.3}",
                g.name, v[0], v[1], v[2], v[3]
            );
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
