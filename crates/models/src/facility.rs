//! Uncapacitated facility location, strong formulation (Cornuéjols, Nemhauser and Wolsey, "The
//! uncapacitated facility location problem", in *Discrete Location Theory*, Wiley 1990). Source:
//! Gurobi modeling-examples `facility_location`.
//!
//! Sites f open at cost F_f; customer c is served from open sites at r per unit distance d_cf
//! (Euclidean), x_cf the share of c served from f.
//!
//!   min Σ_f F_f y_f + Σ_{c,f} r d_cf x_cf   s.t.  Σ_f x_cf = 1,   x_cf ≤ y_f,   0 ≤ x ≤ 1, y binary.
//!
//! Columns: y_f at f, then x_cf at |F| + c |F| + f.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub customers: Vec<(f64, f64)>,
    pub sites: Vec<Site>,
    pub rate: f64,
}

#[derive(Deserialize)]
pub struct Site {
    pub at: (f64, f64),
    pub f: f64,
}

pub fn model(d: &Data) -> Model {
    let nf = d.sites.len();
    let mut m = Model::new(false);
    for s in &d.sites {
        m.bin(s.f);
    }
    for &(cx, cy) in &d.customers {
        for s in &d.sites {
            m.var(0.0, 1.0, d.rate * (cx - s.at.0).hypot(cy - s.at.1));
        }
    }
    for c in 0..d.customers.len() {
        let x = |f: usize| nf + c * nf + f;
        for f in 0..nf {
            m.le([(x(f), 1.0), (f, -1.0)], 0.0);
        }
        m.eq((0..nf).map(|f| (x(f), 1.0)), 1.0);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let nf = d.sites.len();
    let mut text = String::new();
    if !s.x.is_empty() {
        let open: Vec<usize> = (0..nf).filter(|&f| s.x[f] > 0.5).collect();
        let _ = writeln!(text, "cost {:.6}; open sites {open:?}", s.obj);
        for c in 0..d.customers.len() {
            let to: Vec<String> = (0..nf)
                .filter(|&f| s.x[nf + c * nf + f] > 1e-6)
                .map(|f| format!("{f} ({:.2})", s.x[nf + c * nf + f]))
                .collect();
            let _ = writeln!(text, "  customer {c} from {}", to.join(", "));
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
