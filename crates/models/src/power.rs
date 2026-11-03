//! Electrical power generation (unit commitment) 1 and 2: H. P. Williams, *Model Building in
//! Mathematical Programming* (5th ed., 2013), problems "Tariff rates (power generation)" and
//! "Hydro power". Source: Gurobi modeling-examples `electrical_power_generation`.
//!
//! Generator type g (N_g units, output m_g ≤ q ≤ M_g each, costs: hourly base B_g, per MW above
//! minimum c_g, start-up S_g), period p of h_p hours with demand D_p:
//! n_gp units on, s_gp started, q_gp total output.
//!
//!   n_gp ≤ N_g,   m_g n_gp ≤ q_gp ≤ M_g n_gp,   Σ_g q_gp ≥ D_p,   Σ_g M_g n_gp ≥ 1.15 D_p,
//!   n_g0 ≤ n⁰ + s_g0,   n_gp ≤ n_g,p−1 + s_gp;
//!   min Σ h_p (B_g n_gp + c_g (q_gp − m_g n_gp)) + Σ S_g s_gp,   n, s integer.
//!
//! 2 (`hydro`) adds hydro units u (output L_u when on, y_up binary, start z_up, hourly cost and
//! start cost, reservoir drop r_u per hour), pumping π_p and reservoir height H_p:
//!   demand row Σ q + Σ L_u y_up ≥ D_p + π_p,   reserve 1.15 D_p − Σ_u L_u,
//!   H_p = H_p−1 + h_p π_p / 3000 − Σ_u r_u h_p y_up  (cyclic: H_−1 = H_last),
//!   y_u0 ≤ z_u0,   y_up ≤ y_u,p−1 + z_up.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Out, grid};

#[derive(Deserialize)]
pub struct Data {
    pub maxstart0: f64,
    pub generators: Vec<f64>,
    pub period_hours: Vec<f64>,
    pub demand: Vec<f64>,
    pub min_load: Vec<f64>,
    pub max_load: Vec<f64>,
    pub base_cost: Vec<f64>,
    pub per_mw_cost: Vec<f64>,
    pub startup_cost: Vec<f64>,
    pub hydro: Option<Hydro>,
}

#[derive(Deserialize)]
pub struct Hydro {
    pub load: Vec<f64>,
    pub cost: Vec<f64>,
    pub drop: Vec<f64>,
    pub start: Vec<f64>,
    pub pump: f64,
}

/// The model, and the units-on columns n[g][p].
pub fn model(d: &Data) -> (Model, Vec<Vec<usize>>) {
    let (ng, np) = (d.generators.len(), d.demand.len());
    let h = &d.period_hours;
    let mut m = Model::new(false);
    // Output above minimum costs c_g h_p; the minimum's share moves onto n.
    let n = grid(&mut m, ng, np, |m, g, p| {
        m.int(0.0, d.generators[g], h[p] * (d.base_cost[g] - d.per_mw_cost[g] * d.min_load[g]))
    });
    let s = grid(&mut m, ng, np, |m, g, _| m.int(0.0, INF, d.startup_cost[g]));
    let q = grid(&mut m, ng, np, |m, g, p| m.var(0.0, INF, h[p] * d.per_mw_cost[g]));
    let hy = d.hydro.as_ref();
    let nu = hy.map_or(0, |x| x.load.len());
    let y = grid(&mut m, nu, np, |m, u, p| m.bin(hy.unwrap().cost[u] * h[p]));
    let z = grid(&mut m, nu, np, |m, u, _| m.bin(hy.unwrap().start[u]));
    let pump: Vec<usize> = (0..if hy.is_some() { np } else { 0 }).map(|_| m.var(0.0, INF, 0.0)).collect();
    let height: Vec<usize> = (0..pump.len()).map(|_| m.var(0.0, INF, 0.0)).collect();
    let spare: f64 = hy.map_or(0.0, |x| x.load.iter().sum());
    for p in 0..np {
        for g in 0..ng {
            m.ge([(q[g][p], 1.0), (n[g][p], -d.min_load[g])], 0.0);
            m.le([(q[g][p], 1.0), (n[g][p], -d.max_load[g])], 0.0);
        }
        let hydro_out = (0..nu).map(|u| (y[u][p], hy.unwrap().load[u]));
        let pumped = pump.get(p).map(|&j| (j, -1.0));
        m.ge((0..ng).map(|g| (q[g][p], 1.0)).chain(hydro_out).chain(pumped), d.demand[p]);
        m.ge((0..ng).map(|g| (n[g][p], d.max_load[g])), 1.15 * d.demand[p] - spare);
        for g in 0..ng {
            if p == 0 {
                m.le([(n[g][0], 1.0), (s[g][0], -1.0)], d.maxstart0);
            } else {
                m.le([(n[g][p], 1.0), (n[g][p - 1], -1.0), (s[g][p], -1.0)], 0.0);
            }
        }
        if let Some(x) = hy {
            let prev = if p == 0 { np - 1 } else { p - 1 };
            let drop = (0..nu).map(|u| (y[u][p], x.drop[u] * h[p]));
            m.eq([(height[p], 1.0), (height[prev], -1.0), (pump[p], -h[p] / x.pump)].into_iter().chain(drop), 0.0);
            for u in 0..nu {
                let was = if p == 0 { vec![] } else { vec![(y[u][p - 1], -1.0)] };
                m.le([(y[u][p], 1.0), (z[u][p], -1.0)].into_iter().chain(was), 0.0);
            }
        }
    }
    (m, n)
}

pub fn run(d: &Data) -> Out {
    let (m, n) = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "cost {:.2}", s.obj);
        for (g, row) in n.iter().enumerate() {
            let v: Vec<String> = row.iter().map(|&j| format!("{}", s.x[j].round())).collect();
            let _ = writeln!(text, "  type {} units on: {}", g + 1, v.join(" "));
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
