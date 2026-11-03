//! COVID-19 temporary facility location: capacitated facility location with optional temporary
//! sites and a penalised capacity extension. Source: Gurobi modeling-examples
//! `covid19_facility_location` (two demand scenarios, one case file each).
//!
//! Counties c (demand D_c) are served from existing facilities e (capacity K_e) or temporary
//! sites t (capacity K_t if opened, y_t = 1, at cost F); x_cf patients from c to f at r per unit
//! of Euclidean distance d_cf; capacity added beyond K_t costs M per unit (z_t):
//!
//!   Σ_f x_cf = D_c,   Σ_c x_ce ≤ K_e,   Σ_c x_ct − z_t ≤ K_t y_t;
//!   min Σ r d_cf x_cf + F Σ_t y_t + M Σ_t z_t,   x, z ≥ 0, y binary.
//!
//! Columns: y_t, then z_t, then x_cf at 2|T| + |F| c + f (existing facilities first).

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub counties: Vec<County>,
    pub existing: Vec<Site>,
    pub temporary: Vec<Site>,
    pub rate: f64,
    pub fixed: f64,
    pub extra: f64,
}

#[derive(Deserialize)]
pub struct County {
    pub at: (f64, f64),
    pub demand: f64,
}

#[derive(Deserialize)]
pub struct Site {
    pub at: (f64, f64),
    pub cap: f64,
}

pub fn model(d: &Data) -> Model {
    let (ne, nt) = (d.existing.len(), d.temporary.len());
    let f: Vec<&Site> = d.existing.iter().chain(&d.temporary).collect();
    let nf = f.len();
    let mut m = Model::new(false);
    for _ in 0..nt {
        m.bin(d.fixed);
    }
    for _ in 0..nt {
        m.var(0.0, INF, d.extra);
    }
    for c in &d.counties {
        for s in &f {
            m.var(0.0, INF, d.rate * (c.at.0 - s.at.0).hypot(c.at.1 - s.at.1));
        }
    }
    let x = |c: usize, k: usize| 2 * nt + nf * c + k;
    for (c, cy) in d.counties.iter().enumerate() {
        m.eq((0..nf).map(|k| (x(c, k), 1.0)), cy.demand);
    }
    let into = |k: usize| (0..d.counties.len()).map(move |c| (x(c, k), 1.0));
    for (k, s) in d.existing.iter().enumerate() {
        m.le(into(k), s.cap);
    }
    for (t, s) in d.temporary.iter().enumerate() {
        m.le(into(ne + t).chain([(nt + t, -1.0), (t, -s.cap)]), 0.0);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let nt = d.temporary.len();
    let mut text = String::new();
    if !s.x.is_empty() {
        let open: Vec<usize> = (0..nt).filter(|&t| s.x[t] > 0.5).map(|t| d.existing.len() + t + 1).collect();
        let extra: f64 = (0..nt).map(|t| s.x[nt + t]).sum();
        let _ = writeln!(text, "cost {:.4}; temporary sites opened {open:?}; capacity added {extra:.1}", s.obj);
    }
    Out::of(&m, &s, vec![s.obj], text)
}
