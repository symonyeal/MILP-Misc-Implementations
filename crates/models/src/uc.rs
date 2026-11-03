//! Power generation: a one-day unit commitment for Georgia's plants (1 July 2011). Source: Gurobi
//! modeling-examples `power_generation` (`small_plant_data`).
//!
//! Plants i with capacity c_i, minimum share m_i when on, ramp share r_i, fuel cost f_i and health
//! cost a_ih per MWh, operating cost o_i per hour on, start-up s_i and shut-down t_i; hours h with
//! demand d_h. Columns: output z_ih ≥ 0, on u_ih, start v_ih, stop w_ih binary:
//!
//!   Σ_i z_ih = d_h,   m_i c_i u_ih ≤ z_ih ≤ c_i u_ih,   z_ih ≥ m_i c_i (nuclear),
//!   |z_ih − z_i,h−1| ≤ r_i c_i,   v_ih ≤ u_ih,   w_ih ≤ 1 − u_ih,   v_ih − w_ih = u_ih − u_i,h−1;
//!   min Σ (f_i + a_ih) z_ih + o_i u_ih + s_i v_ih + t_i w_ih.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Plant {
    pub name: String,
    pub nuclear: bool,
    pub cap: f64,
    pub min: f64,
    pub ramp: f64,
    pub fuel: f64,
    pub oper: f64,
    pub start: f64,
    pub stop: f64,
    /// a_ih per hour.
    pub health: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Data {
    pub plants: Vec<Plant>,
    pub demand: Vec<f64>,
}

/// The model and z, u indexed [plant][hour].
pub fn model(d: &Data) -> (Model, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let nh = d.demand.len();
    let mut m = Model::new(false);
    let (mut z, mut u) = (Vec::new(), Vec::new());
    for p in &d.plants {
        let lo = if p.nuclear { p.min * p.cap } else { 0.0 };
        let zi: Vec<usize> = (0..nh).map(|h| m.var(lo, INF, p.fuel + p.health[h])).collect();
        let ui: Vec<usize> = (0..nh).map(|_| m.bin(p.oper)).collect();
        let vi: Vec<usize> = (0..nh).map(|_| m.bin(p.start)).collect();
        let wi: Vec<usize> = (0..nh).map(|_| m.bin(p.stop)).collect();
        for h in 0..nh {
            m.le([(zi[h], 1.0), (ui[h], -p.cap)], 0.0);
            m.ge([(zi[h], 1.0), (ui[h], -p.min * p.cap)], 0.0);
            m.le([(vi[h], 1.0), (ui[h], -1.0)], 0.0);
            m.le([(wi[h], 1.0), (ui[h], 1.0)], 1.0);
            if h > 0 {
                m.row(-p.ramp * p.cap, [(zi[h], 1.0), (zi[h - 1], -1.0)], p.ramp * p.cap);
                m.eq([(vi[h], 1.0), (wi[h], -1.0), (ui[h], -1.0), (ui[h - 1], 1.0)], 0.0);
            }
        }
        z.push(zi);
        u.push(ui);
    }
    for (h, &dh) in d.demand.iter().enumerate() {
        m.eq(z.iter().map(|zi| (zi[h], 1.0)), dh);
    }
    (m, z, u)
}

pub fn run(d: &Data) -> Out {
    let (m, z, u) = model(d);
    let s = m.solve();
    let mut text = String::new();
    let _ = writeln!(text, "cost {:.2}", s.obj);
    for (p, (zi, ui)) in d.plants.iter().zip(z.iter().zip(&u)) {
        let on = ui.iter().filter(|&&j| s.x[j] > 0.5).count();
        let e: f64 = zi.iter().map(|&j| s.x[j]).sum();
        let _ = writeln!(text, "  {:<15} {on:>2} h on, {e:>9.1} MWh", p.name);
    }
    Out::of(&m, &s, vec![s.obj], text)
}
