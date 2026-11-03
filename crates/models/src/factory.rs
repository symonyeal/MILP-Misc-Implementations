//! Factory Planning I and II: H. P. Williams, *Model Building in Mathematical Programming*
//! (5th ed., Wiley 2013), problems "Factory planning 1" and "Factory planning 2". Source: Gurobi
//! modeling-examples `factory_planning`.
//!
//! Month t, product p, machine type m: make y_tp, sell q_tp ≤ Q_tp, hold s_tp ≤ s̄; r_tm machines of
//! type m are down; k_m are installed, each giving H hours a month; a_mp hours per unit.
//!
//!   max Σ_t Σ_p (π_p q_tp − h s_tp)
//!   s_{t−1,p} + y_tp = q_tp + s_tp  (s_{−1,p} = 0),   s_{T−1,p} = target,
//!   Σ_p a_mp y_tp ≤ H (k_m − r_tm).
//!
//! I: r_tm is fixed at the data's `down`. II: r_tm ∈ ℤ ∩ [0, R_m] with Σ_t r_tm = R_m (`maint`),
//! so the model also schedules maintenance. Capacity rows hold H r_tm on the left.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Out, grid, stock};

#[derive(Deserialize)]
pub struct Data {
    pub months: Vec<String>,
    pub products: Vec<Product>,
    pub machines: Vec<Machine>,
    pub time: Vec<Vec<f64>>,
    pub sales: Vec<Vec<f64>>,
    pub hold: f64,
    pub smax: f64,
    pub target: f64,
    pub hours: f64,
    pub down: Option<Vec<Vec<f64>>>,
    pub maint: Option<Vec<f64>>,
}

#[derive(Deserialize)]
pub struct Product {
    pub name: String,
    pub profit: f64,
}

#[derive(Deserialize)]
pub struct Machine {
    pub name: String,
    pub installed: f64,
}

/// Column indices by [t][p] (y, q, s) and [t][m] (r).
pub struct Ix {
    pub y: Vec<Vec<usize>>,
    pub q: Vec<Vec<usize>>,
    pub s: Vec<Vec<usize>>,
    pub r: Vec<Vec<usize>>,
}

pub fn model(d: &Data) -> (Model, Ix) {
    assert!(d.down.is_some() != d.maint.is_some(), "factory data need exactly one of down, maint");
    let (nt, np, nm) = (d.months.len(), d.products.len(), d.machines.len());
    let mut m = Model::new(true);
    let y = grid(&mut m, nt, np, |m, _, _| m.var(0.0, INF, 0.0));
    let q = grid(&mut m, nt, np, |m, t, p| m.var(0.0, d.sales[t][p], d.products[p].profit));
    let s = grid(&mut m, nt, np, |m, _, _| m.var(0.0, d.smax, -d.hold));
    let r = grid(&mut m, nt, nm, |m, t, k| match (&d.down, &d.maint) {
        (Some(dn), _) => m.var(dn[t][k], dn[t][k], 0.0),
        (_, Some(mt)) => m.int(0.0, mt[k], 0.0),
        _ => unreachable!(),
    });
    for p in 0..np {
        let col = |x: &Vec<Vec<usize>>| (0..nt).map(|t| x[t][p]).collect::<Vec<_>>();
        stock(&mut m, 0.0, &col(&y), &col(&q), &col(&s), d.target);
    }
    for t in 0..nt {
        for (k, mc) in d.machines.iter().enumerate() {
            let a = (0..np).map(|p| (y[t][p], d.time[k][p]));
            m.le(a.chain([(r[t][k], d.hours)]), d.hours * mc.installed);
        }
    }
    if let Some(mt) = &d.maint {
        for k in 0..nm {
            m.eq((0..nt).map(|t| (r[t][k], 1.0)), mt[k]);
        }
    }
    (m, Ix { y, q, s, r })
}

pub fn run(d: &Data) -> Out {
    let (m, ix) = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "profit {:.2}", s.obj);
        let head: Vec<&str> = d.products.iter().map(|p| p.name.as_str()).collect();
        for (lab, x) in [("make", &ix.y), ("sell", &ix.q), ("store", &ix.s)] {
            let _ = writeln!(text, "{lab:6} {}", head.iter().map(|h| format!("{h:>8}")).collect::<String>());
            for (t, row) in x.iter().enumerate() {
                let _ = writeln!(
                    text,
                    "{:6} {}",
                    d.months[t],
                    row.iter().map(|&j| format!("{:8.1}", s.x[j])).collect::<String>()
                );
            }
        }
        if d.maint.is_some() {
            let _ =
                writeln!(text, "down   {}", d.machines.iter().map(|k| format!("{:>10}", k.name)).collect::<String>());
            for (t, row) in ix.r.iter().enumerate() {
                let _ = writeln!(
                    text,
                    "{:6} {}",
                    d.months[t],
                    row.iter().map(|&j| format!("{:10}", s.x[j].round() + 0.0)).collect::<String>()
                );
            }
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
