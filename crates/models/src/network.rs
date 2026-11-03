//! Supply network design: H. P. Williams, *Model Building in Mathematical Programming* (5th ed.,
//! 2013), problems "Distribution 1" and "Depot location (distribution 2)". Source: Gurobi
//! modeling-examples `supply_network_design` 1 and 2.
//!
//! Goods flow x_a on arcs a = (i, j) at k_a a unit, from factories (supply S) through depots
//! (throughput C) or directly to customers (demand D). Depot d opens (o_d = 1) at F_d; expansion e
//! adds E_e throughput at depot d(e) for g_e; at most N depots open.
//!
//!   min c0 + Σ_a k_a x_a + Σ_d F_d o_d + Σ_e g_e w_e
//!   out(i) ≤ S_i,   in(j) = D_j,   in(d) = out(d),   in(d) ≤ C_d o_d + Σ_{e: d(e)=d} E_e w_e,
//!   Σ_d o_d ≤ N,   x ≥ 0,   o, w binary.
//!
//! A depot without an open cost has no o_d: it is open (design 1). `must` fixes o_d = 1. The
//! notebook bounds depot outflow rather than inflow for all but Birmingham; with in(d) = out(d)
//! the rows are the same.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Data {
    pub factories: Vec<Factory>,
    pub depots: Vec<Depot>,
    pub customers: Vec<Customer>,
    pub arcs: Vec<(String, String, f64)>,
    #[serde(default)]
    pub expand: Vec<Expand>,
    pub max_open: Option<f64>,
    #[serde(default)]
    pub c0: f64,
}

#[derive(Deserialize)]
pub struct Factory {
    pub name: String,
    pub supply: f64,
}

#[derive(Deserialize)]
pub struct Depot {
    pub name: String,
    pub cap: f64,
    pub open: Option<f64>,
    #[serde(default)]
    pub must: bool,
}

#[derive(Deserialize)]
pub struct Customer {
    pub name: String,
    pub demand: f64,
}

#[derive(Deserialize)]
pub struct Expand {
    pub depot: String,
    pub cap: f64,
    pub cost: f64,
}

/// Column indices beyond x_a (column a): o by depot (None when always open), w by expansion.
pub struct Ix {
    pub o: Vec<Option<usize>>,
    pub w: Vec<usize>,
}

pub fn model(d: &Data) -> (Model, Ix) {
    let nodes = d
        .factories
        .iter()
        .map(|f| f.name.as_str())
        .chain(d.depots.iter().map(|x| x.name.as_str()))
        .chain(d.customers.iter().map(|c| c.name.as_str()));
    let names = Names::new(nodes);
    let (nf, nd) = (d.factories.len(), d.depots.len());
    let ends: Vec<(usize, usize)> = d.arcs.iter().map(|(i, j, _)| (names.at(i), names.at(j))).collect();
    let mut m = Model::new(false);
    m.c0 = d.c0;
    for (_, _, k) in &d.arcs {
        m.var(0.0, INF, *k);
    }
    let o: Vec<Option<usize>> =
        d.depots.iter().map(|x| x.open.map(|f| m.int(if x.must { 1.0 } else { 0.0 }, 1.0, f))).collect();
    let dn = Names::new(d.depots.iter().map(|x| x.name.as_str()));
    let w: Vec<usize> = d.expand.iter().map(|e| m.bin(e.cost)).collect();
    let flow = |v: usize, into: bool, s: f64| {
        ends.iter().enumerate().filter(move |(_, e)| if into { e.1 == v } else { e.0 == v }).map(move |(a, _)| (a, s))
    };
    for (i, f) in d.factories.iter().enumerate() {
        m.le(flow(i, false, 1.0), f.supply);
    }
    for (c, cu) in d.customers.iter().enumerate() {
        m.eq(flow(nf + nd + c, true, 1.0), cu.demand);
    }
    for (k, dp) in d.depots.iter().enumerate() {
        let v = nf + k;
        m.eq(flow(v, true, 1.0).chain(flow(v, false, -1.0)), 0.0);
        let ex = d.expand.iter().enumerate().filter(|(_, e)| dn.at(&e.depot) == k).map(|(e, x)| (w[e], -x.cap));
        match o[k] {
            Some(ok) => m.le(flow(v, true, 1.0).chain([(ok, -dp.cap)]).chain(ex), 0.0),
            None => m.le(flow(v, true, 1.0).chain(ex), dp.cap),
        };
    }
    if let Some(n) = d.max_open {
        m.le(o.iter().flatten().map(|&j| (j, 1.0)), n);
    }
    (m, Ix { o, w })
}

pub fn run(d: &Data) -> Out {
    let (m, ix) = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "cost {}", s.obj);
        let open: Vec<&str> = d
            .depots
            .iter()
            .zip(&ix.o)
            .filter(|(_, o)| o.is_none_or(|j| s.x[j] > 0.5))
            .map(|(x, _)| x.name.as_str())
            .collect();
        let _ = writeln!(text, "open depots {}", open.join(" "));
        for (e, &j) in d.expand.iter().zip(&ix.w) {
            let _ = writeln!(text, "expand {}: {}", e.depot, s.x[j] > 0.5);
        }
        for ((i, j, _), x) in d.arcs.iter().zip(&s.x) {
            if *x > 1e-6 {
                let _ = writeln!(text, "  {i:>11} -> {j:<11} {x}");
            }
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
