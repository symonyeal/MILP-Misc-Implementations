//! Car rental: fleet size and a steady-state weekly plan. H. P. Williams, *Model Building in
//! Mathematical Programming* (5th ed., 2013), problems "Car rental 1" and "Car rental 2". Source:
//! Gurobi modeling-examples `car_rental_1`, `car_rental_2`.
//!
//! Depots d (R the repair depots), days t = 0..5 (Monday..Saturday; arithmetic mod 6, so Saturday's
//! 1-day rentals return on Monday), rental lengths r = 1..3 with shares π_r. A car rented at d
//! returns to e with share p_de, undamaged with share ρ, damaged with 1 − ρ. Columns per (d, t):
//! u, w undamaged and damaged cars at the start of the day, h hired (h ≤ demand), ū, w̄ left
//! over, y repaired (d ∈ R); transfers v_det undamaged and z_det damaged (d ≠ e, arriving t + 1);
//! the fleet n; expansion binaries x_dk.
//!
//!   undamaged in   Σ_{e,r} ρ p_ed π_r h_e,t−r + Σ_e v_ed,t−1 + y_d,t−1 + ū_d,t−1 = u_dt
//!   undamaged out  h_dt + Σ_e v_det + ū_dt = u_dt
//!   damaged in     Σ_{e,r} (1 − ρ) p_ed π_r h_e,t−r + [d ∈ R] Σ_e z_ed,t−1 + w̄_d,t−1 = w_dt
//!   damaged out    [d ∈ R] y_dt + Σ_{e ∈ R} z_det + w̄_dt = w_dt
//!   repair         y_dt ≤ κ_d + s Σ_k x_dk,   x_dk ≥ x_d,k+1,   Σ x ≤ K
//!   fleet          n = Σ_d (w_d2 + u_d2 + Σ_{t,r: 0 < (2 − t) mod 6 < r} π_r h_dt)  (Wednesday morning)
//!   max Σ_{d,t} h_dt Σ_r π_r Σ_e p_de (price_r(d = e) − c_r + f)
//!       − Σ c_de (v_det + z_det) − o n − Σ F_dk x_dk.
//!
//! The notebooks' damaged-out row at a repair depot sums z_det over e ∉ R, against their own
//! formulation, which sums over e ∈ R: those cars leave the depot and never arrive, and transfers
//! into another repair depot arrive without leaving. `model(d, true)` keeps that row (its optima are
//! Gurobi's); `run` solves the corrected one.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Out, grid};

#[derive(Deserialize)]
pub struct Data {
    pub depots: Vec<String>,
    /// Repair depots, as indices into `depots`.
    pub repair: Vec<usize>,
    /// demand[d][t].
    pub demand: Vec<Vec<f64>>,
    /// κ_d, cars repaired per day.
    pub cap: Vec<f64>,
    /// π_r, c_r, same-depot and other-depot prices, for r = 1..3.
    pub share: Vec<f64>,
    pub cost: Vec<f64>,
    pub same: Vec<f64>,
    pub other: Vec<f64>,
    /// p[d][e], transfer cost c[d][e].
    pub p: Vec<Vec<f64>>,
    pub c: Vec<Vec<f64>>,
    /// ρ, f (expected damage excess per rental), o (weekly cost of owning a car).
    pub undamaged: f64,
    pub fee: f64,
    pub own: f64,
    /// F_dk: the costs of depot d's successive expansions, each adding `step` repairs a day; at
    /// most `kmax` in all.
    #[serde(default)]
    pub expand: Vec<Vec<f64>>,
    #[serde(default)]
    pub step: f64,
    #[serde(default)]
    pub kmax: f64,
}

const T: usize = 6;

/// The model; `nb` keeps the notebooks' damaged-out row (see the header). Columns: n first.
pub fn model(d: &Data, nb: bool) -> Model {
    let nd = d.depots.len();
    let rd = |k: usize| d.repair.contains(&k);
    let mut m = Model::new(true);
    let n = m.var(0.0, INF, -d.own);
    let rev = |k: usize| -> f64 {
        (0..3)
            .map(|r| {
                let pr = (0..nd).map(|e| d.p[k][e] * if e == k { d.same[r] } else { d.other[r] }).sum::<f64>();
                d.share[r] * (pr - d.cost[r] + d.fee)
            })
            .sum()
    };
    let h = grid(&mut m, nd, T, |m, k, t| m.var(0.0, d.demand[k][t], rev(k)));
    let u = grid(&mut m, nd, T, |m, _, _| m.var(0.0, INF, 0.0));
    let w = grid(&mut m, nd, T, |m, _, _| m.var(0.0, INF, 0.0));
    let ul = grid(&mut m, nd, T, |m, _, _| m.var(0.0, INF, 0.0));
    let wl = grid(&mut m, nd, T, |m, _, _| m.var(0.0, INF, 0.0));
    let y = grid(&mut m, nd, T, |m, k, _| m.var(0.0, if rd(k) { INF } else { 0.0 }, 0.0));
    let pairs: Vec<(usize, usize)> =
        (0..nd).flat_map(|a| (0..nd).filter(move |&b| b != a).map(move |b| (a, b))).collect();
    let v = grid(&mut m, pairs.len(), T, |m, k, _| m.var(0.0, INF, -d.c[pairs[k].0][pairs[k].1]));
    let z = grid(&mut m, pairs.len(), T, |m, k, _| m.var(0.0, INF, -d.c[pairs[k].0][pairs[k].1]));
    let x: Vec<Vec<usize>> = d.expand.iter().map(|f| f.iter().map(|&f| m.bin(-f)).collect()).collect();
    let prev = |t: usize| (t + T - 1) % T;
    // Returns at depot k on day t, share q of each: Σ_{e,r} q p_ek π_r h_e,t−r.
    let hr = &h;
    let back = |k: usize, t: usize, q: f64| -> Vec<(usize, f64)> {
        (0..nd).flat_map(|e| (0..3).map(move |r| (hr[e][(t + T - 1 - r) % T], q * d.p[e][k] * d.share[r]))).collect()
    };
    for k in 0..nd {
        for t in 0..T {
            let into = |b: usize| pairs.iter().enumerate().filter(move |(_, p)| p.1 == b).map(|(j, _)| j);
            let out = |a: usize| pairs.iter().enumerate().filter(move |(_, p)| p.0 == a);
            let mut r = back(k, t, d.undamaged);
            r.extend(into(k).map(|j| (v[j][prev(t)], 1.0)));
            r.extend([(y[k][prev(t)], 1.0), (ul[k][prev(t)], 1.0), (u[k][t], -1.0)]);
            m.eq(r, 0.0);
            let mut r: Vec<(usize, f64)> = out(k).map(|(j, _)| (v[j][t], 1.0)).collect();
            r.extend([(h[k][t], 1.0), (ul[k][t], 1.0), (u[k][t], -1.0)]);
            m.eq(r, 0.0);
            let mut r = back(k, t, 1.0 - d.undamaged);
            if rd(k) {
                r.extend(into(k).map(|j| (z[j][prev(t)], 1.0)));
            }
            r.extend([(wl[k][prev(t)], 1.0), (w[k][t], -1.0)]);
            m.eq(r, 0.0);
            let to = |e: usize| if nb && rd(k) { !rd(e) } else { rd(e) };
            let mut r: Vec<(usize, f64)> = out(k).filter(|(_, p)| to(p.1)).map(|(j, _)| (z[j][t], 1.0)).collect();
            r.extend([(y[k][t], 1.0), (wl[k][t], 1.0), (w[k][t], -1.0)]);
            m.eq(r, 0.0);
            if rd(k) {
                let xs = x.get(k).into_iter().flatten().map(|&j| (j, -d.step));
                m.le([(y[k][t], 1.0)].into_iter().chain(xs), d.cap[k]);
            }
        }
        for f in x.get(k).into_iter().flat_map(|f| f.windows(2)) {
            m.ge([(f[0], 1.0), (f[1], -1.0)], 0.0);
        }
    }
    if x.iter().any(|f| !f.is_empty()) {
        m.le(x.iter().flatten().map(|&j| (j, 1.0)), d.kmax);
    }
    let mut r = vec![(n, -1.0)];
    for k in 0..nd {
        r.extend([(u[k][2], 1.0), (w[k][2], 1.0)]);
        for t in 0..T {
            let out = (0..3).filter(|&r| (1..=r).contains(&((2 + T - t) % T))).map(|r| d.share[r]).sum::<f64>();
            if out > 0.0 {
                r.push((h[k][t], out));
            }
        }
    }
    m.eq(r, 0.0);
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d, false);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "profit {:.2}, fleet {:.2}", s.obj, s.x[0]);
        let nx: usize = d.expand.iter().map(Vec::len).sum();
        let x = &s.x[s.x.len() - nx..];
        let mut k = 0;
        for (j, f) in d.expand.iter().enumerate() {
            let on = x[k..k + f.len()].iter().filter(|&&v| v > 0.5).count();
            k += f.len();
            if on > 0 {
                let _ = writeln!(text, "  {}: {on} expansion(s)", d.depots[j]);
            }
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the notebooks' row, the optima are Gurobi's.
    #[test]
    fn notebook_rows_give_gurobi_optima() {
        for f in ["car_rental_1", "car_rental_2"] {
            let p = format!("{}/../../data/{f}.json", env!("CARGO_MANIFEST_DIR"));
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
            let z = v.get("gurobi").unwrap_or(&v["oracle"])[0].as_f64().unwrap();
            let s = model(&serde_json::from_value(v["data"].clone()).unwrap(), true).solve();
            assert!((s.obj - z).abs() <= 1e-6 * z.abs(), "{f}: {} against {z}", s.obj);
        }
    }
}
