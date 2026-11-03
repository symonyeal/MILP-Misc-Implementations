//! Decentralization: H. P. Williams, *Model Building in Mathematical Programming* (5th ed.,
//! 2013), problem "Decentralization". Source: Gurobi modeling-examples `decentralization_planning`.
//!
//! Place each department d in one city c (δ_dc = 1), at most K per city, gaining B_dc but paying
//! C_dc,d'c' for each pair of placed departments that communicate:
//!
//!   max Σ B_dc δ_dc − Σ C_dc,d'c' δ_dc δ_d'c'   s.t.  Σ_c δ_dc = 1,   Σ_d δ_dc ≤ K,   δ binary.
//!
//! A binary quadratic program, as the notebook states it; SCIP solves it (Williams linearises the
//! products with extra binaries). Columns: δ_dc at |C| d + c.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Data {
    pub departments: Vec<String>,
    pub cities: Vec<String>,
    pub benefit: Vec<(String, String, f64)>,
    pub comm: Vec<(String, String, String, String, f64)>,
    pub max_per_city: f64,
}

pub fn model(d: &Data) -> Model {
    let (dn, cn) =
        (Names::new(d.departments.iter().map(String::as_str)), Names::new(d.cities.iter().map(String::as_str)));
    let (nd, nc) = (d.departments.len(), d.cities.len());
    let x = |a: &str, c: &str| nc * dn.at(a) + cn.at(c);
    let mut m = Model::new(true);
    for _ in 0..nd * nc {
        m.bin(0.0);
    }
    for (a, c, b) in &d.benefit {
        m.c[x(a, c)] += b;
    }
    for (a, c, e, f, v) in &d.comm {
        m.qobj(x(a, c), x(e, f), -v);
    }
    for k in 0..nd {
        m.eq((0..nc).map(|c| (nc * k + c, 1.0)), 1.0);
    }
    for c in 0..nc {
        m.le((0..nd).map(|k| (nc * k + c, 1.0)), d.max_per_city);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let nc = d.cities.len();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "net benefit {:.4} (thousands of pounds)", s.obj);
        for (k, dep) in d.departments.iter().enumerate() {
            let c = (0..nc).find(|&c| s.x[nc * k + c] > 0.5).map_or("-", |c| d.cities[c].as_str());
            let _ = writeln!(text, "  {dep}: {c}");
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
