//! Agricultural pricing: H. P. Williams, *Model Building in Mathematical Programming* (5th ed.,
//! 2013), problem "Agricultural pricing". Source: Gurobi modeling-examples `agricultural_pricing`.
//!
//! Set prices p_d of dairy products d to maximise revenue Σ_d q_d p_d, where demand q_d answers
//! price through elasticities around last year's (q⁰_d, p⁰_d):
//!
//!   (q_d − q⁰_d)/q⁰_d = −e_d (p_d − p⁰_d)/p⁰_d + Σ_{(d,c) cross} e_dc (p_c − p⁰_c)/p⁰_c,
//!   Σ_d a_cd q_d ≤ K_c (components c: fat, dry matter),   Σ_d q⁰_d p_d ≤ I (price index),
//!   p, q ≥ 0.
//!
//! The objective is bilinear and indefinite as written (Williams substitutes q to make it a
//! concave QP in p); SCIP solves it as stated, globally. The rows bound p (index row) and q
//! (component rows), which spatial branching needs. Columns: q_d at d, then p_d at |D| + d.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Data {
    pub products: Vec<Product>,
    pub components: Vec<Component>,
    pub cross: Vec<(String, String, f64)>,
    pub index: f64,
}

#[derive(Deserialize)]
pub struct Product {
    pub name: String,
    pub q0: f64,
    pub p0: f64,
    pub e: f64,
}

#[derive(Deserialize)]
pub struct Component {
    pub name: String,
    pub cap: f64,
    pub share: Vec<f64>,
}

pub fn model(d: &Data) -> Model {
    let nd = d.products.len();
    let names = Names::new(d.products.iter().map(|p| p.name.as_str()));
    let mut m = Model::new(true);
    for _ in 0..2 * nd {
        m.var(0.0, INF, 0.0);
    }
    let p = |k: usize| nd + k;
    for k in 0..nd {
        m.qobj(k, p(k), 1.0);
    }
    for c in &d.components {
        m.le((0..nd).map(|k| (k, c.share[k])), c.cap);
    }
    m.le(d.products.iter().enumerate().map(|(k, x)| (p(k), x.q0)), d.index);
    // q/q⁰ + e p/p⁰ − Σ e_c p_c/p⁰_c = 1 + e − Σ e_c, the elasticity row with constants collected.
    for (k, x) in d.products.iter().enumerate() {
        let cr: Vec<(usize, f64)> =
            d.cross.iter().filter(|(a, _, _)| *a == x.name).map(|(_, c, e)| (names.at(c), *e)).collect();
        let t = [(k, 1.0 / x.q0), (p(k), x.e / x.p0)]
            .into_iter()
            .chain(cr.iter().map(|&(c, e)| (p(c), -e / d.products[c].p0)));
        m.eq(t, 1.0 + x.e - cr.iter().map(|c| c.1).sum::<f64>());
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let nd = d.products.len();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "revenue {:.6}", s.obj);
        for (k, x) in d.products.iter().enumerate() {
            let _ = writeln!(
                text,
                "  {:8} price {:.4} (was {}), demand {:.4} (was {})",
                x.name,
                s.x[nd + k],
                x.p0,
                s.x[k],
                x.q0
            );
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
