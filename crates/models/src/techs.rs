//! Technician routing and scheduling: assign jobs to skilled technicians, route each from its
//! depot, and time the visits against soft windows and due times. Source: Gurobi
//! modeling-examples `technician_routing_scheduling` (scenarios 0 and 3).
//!
//! Technicians k with capacity Q_k (minutes) and depot o_k; customers j at location ℓ_j with job
//! duration p_j, priority w_j, window [a_j, b_j], due time e_j, and the technicians K_j able to do
//! the job; travel times τ. Columns: x_jk, u_k, arcs y_ilk on locations, g_j (job dropped), binary;
//! t_ℓ ∈ [0, 600] the start at location ℓ, lateness z_j, window violations α_j, β_j ≥ 0:
//!
//!   Σ_{k ∈ K_j} x_jk + g_j = 1,   Σ_k x_jk ≤ 1,   Σ_j p_j x_jk + Σ τ_il y_ilk ≤ Q_k u_k,
//!   Σ_i y_i,ℓj,k = Σ_l y_ℓj,l,k = x_jk,   Σ_j y_ℓj,o_k,k = Σ_j y_o_k,ℓj,k = u_k,
//!   y on another technician's depot = 0,
//!   t_ℓj ≥ t_ℓi + p_i + τ − M (1 − Σ_k y_ℓi,ℓj,k),   t_ℓj ≥ t_o + τ − M (1 − Σ_k y_o,ℓj,k),
//!   t_ℓj + α_j ≥ a_j,   t_ℓj − β_j ≤ b_j,   z_j ≥ t_ℓj + p_j − e_j;
//!   min Σ w_j z_j + 0.01 P Σ w_j (α_j + β_j) + P Σ w_j g_j,   P = 6100.
//!
//! The time rows forbid cycles among customers (p_j > 0), so no subtour rows are needed. Times
//! are per location, which is exact here: no two customers share one.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Tech {
    pub name: String,
    pub cap: f64,
    pub depot: String,
}

#[derive(Deserialize)]
pub struct Customer {
    pub name: String,
    pub loc: String,
    pub duration: f64,
    pub priority: f64,
    /// Indices of the technicians able to do the job.
    pub by: Vec<usize>,
    pub start: f64,
    pub end: f64,
    pub due: f64,
}

#[derive(Deserialize)]
pub struct Data {
    pub locations: Vec<String>,
    /// τ[ℓ][ℓ'] in minutes.
    pub time: Vec<Vec<f64>>,
    pub techs: Vec<Tech>,
    pub customers: Vec<Customer>,
}

const P: f64 = 6100.0;
const HORIZON: f64 = 600.0;

/// The model, x[j][k] and t[ℓ].
pub fn model(d: &Data) -> (Model, Vec<Vec<usize>>, Vec<usize>) {
    let at = Names::new(d.locations.iter().map(String::as_str));
    let (nl, nk) = (d.locations.len(), d.techs.len());
    let loc: Vec<usize> = d.customers.iter().map(|c| at.at(&c.loc)).collect();
    let dep: Vec<usize> = d.techs.iter().map(|k| at.at(&k.depot)).collect();
    let mut depots = dep.clone();
    depots.sort_unstable();
    depots.dedup();
    let mut m = Model::new(false);
    let x: Vec<Vec<usize>> = d.customers.iter().map(|_| (0..nk).map(|_| m.bin(0.0)).collect()).collect();
    let u: Vec<usize> = (0..nk).map(|_| m.bin(0.0)).collect();
    let y: Vec<Vec<Vec<usize>>> =
        (0..nl).map(|_| (0..nl).map(|_| (0..nk).map(|_| m.bin(0.0)).collect()).collect()).collect();
    for (k, &o) in dep.iter().enumerate() {
        for &dd in depots.iter().filter(|&&dd| dd != o) {
            for row in y.iter().take(nl) {
                m.u[row[dd][k]] = 0.0;
            }
            for &c in y[dd].iter().map(|r| &r[k]) {
                m.u[c] = 0.0;
            }
        }
    }
    let t: Vec<usize> = (0..nl).map(|_| m.var(0.0, HORIZON, 0.0)).collect();
    let mut z = Vec::new();
    for (j, c) in d.customers.iter().enumerate() {
        z.push(m.var(0.0, INF, c.priority));
        let a = m.var(0.0, INF, 0.01 * P * c.priority);
        let b = m.var(0.0, INF, 0.01 * P * c.priority);
        let g = m.bin(P * c.priority);
        m.eq(c.by.iter().map(|&k| (x[j][k], 1.0)).chain([(g, 1.0)]), 1.0);
        m.le(x[j].iter().map(|&c| (c, 1.0)), 1.0);
        m.ge([(t[loc[j]], 1.0), (a, 1.0)], c.start);
        m.le([(t[loc[j]], 1.0), (b, -1.0)], c.end);
        m.ge([(z[j], 1.0), (t[loc[j]], -1.0)], c.duration - c.due);
    }
    for (k, tk) in d.techs.iter().enumerate() {
        let mut r: Vec<(usize, f64)> = d.customers.iter().enumerate().map(|(j, c)| (x[j][k], c.duration)).collect();
        for i in 0..nl {
            for l in 0..nl {
                r.push((y[i][l][k], d.time[i][l]));
            }
        }
        r.push((u[k], -tk.cap));
        m.le(r, 0.0);
        for (j, &l) in loc.iter().enumerate() {
            m.eq((0..nl).map(|i| (y[i][l][k], 1.0)).chain([(x[j][k], -1.0)]), 0.0);
            m.eq((0..nl).map(|i| (y[l][i][k], 1.0)).chain([(x[j][k], -1.0)]), 0.0);
        }
        let o = dep[k];
        m.eq(loc.iter().map(|&l| (y[l][o][k], 1.0)).chain([(u[k], -1.0)]), 0.0);
        m.eq(loc.iter().map(|&l| (y[o][l][k], 1.0)).chain([(u[k], -1.0)]), 0.0);
    }
    for (i, ci) in d.customers.iter().enumerate() {
        for (j, _) in d.customers.iter().enumerate() {
            let (li, lj) = (loc[i], loc[j]);
            let big = HORIZON + ci.duration + d.time[li][lj];
            // t_j − t_i − M Σ_k y ≥ p_i + τ − M.
            let r = [(t[lj], 1.0), (t[li], -1.0)].into_iter().chain((0..nk).map(|k| (y[li][lj][k], -big)));
            m.ge(r, ci.duration + d.time[li][lj] - big);
        }
    }
    for &o in &depots {
        for &lj in &loc {
            let big = HORIZON + d.time[o][lj];
            let r = [(t[lj], 1.0), (t[o], -1.0)].into_iter().chain((0..nk).map(|k| (y[o][lj][k], -big)));
            m.ge(r, d.time[o][lj] - big);
        }
    }
    (m, x, t)
}

pub fn run(d: &Data) -> Out {
    let (m, x, t) = model(d);
    let s = m.solve();
    let at = Names::new(d.locations.iter().map(String::as_str));
    let mut text = String::new();
    let _ = writeln!(text, "penalty {:.2}", s.obj);
    for (j, c) in d.customers.iter().enumerate() {
        let who = (0..d.techs.len()).find(|&k| s.x[x[j][k]] > 0.5);
        let _ = match who {
            Some(k) => {
                writeln!(text, "  {} at {}: {} from t = {:.0}", c.name, c.loc, d.techs[k].name, s.x[t[at.at(&c.loc)]])
            }
            None => writeln!(text, "  {} at {}: nobody", c.name, c.loc),
        };
    }
    Out::of(&m, &s, vec![s.obj], text)
}
