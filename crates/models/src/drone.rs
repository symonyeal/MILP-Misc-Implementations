//! Drone network: which bases get the defibrillator drones, and which base answers each cardiac
//! arrest location, each base an M/G/1 queue. Source: Gurobi modeling-examples `drone_network`
//! (Virginia Beach, June 2019).
//!
//! Locations i with call rate λ_i per hour, bases j, distance δ_ij (m), speed v (km/h), service
//! time s_ij = μ + 2 δ_ij/(1000 v) hours with maintenance μ; at most N bases. Columns: x_j base
//! open, y_ij assignment, binary; V_j, ν_ij continuous:
//!
//!   y_ij ≤ x_j,   Σ_j x_j ≤ N,   Σ_j y_ij = 1,   ρ_j = Σ_i λ_i s_ij y_ij ≤ 1,
//!   V_j = Σ_i λ_i s_ij² y_ij + Σ_i λ_i s_ij ν_ij,   ν_ij = V_j y_ij (McCormick, V_j ≤ V̄ = 2 Σ s),
//!   min (1/Λ) Σ_ij λ_i (½ ν_ij + τ_ij y_ij),   τ_ij = δ_ij/(1000 v),   Λ = Σ λ_i.
//!
//! V_j = Σ λ s² y / (1 − ρ_j) is twice the Pollaczek–Khinchine mean wait W_j, so Σ_i λ_i ν_ij / 2
//! = Λ_j W_j and the objective is the mean wait plus the mean flight time, in hours. The notebook's
//! τ_ij is δ_ij / v, metres over km/h, a thousand times the flight time, so its optimum (Gurobi's
//! 37.32) is a nearest-base assignment that ignores the queues; `notebook` keeps that term.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub bases: Vec<String>,
    /// λ_i per hour.
    pub rate: Vec<f64>,
    /// δ_ij in metres.
    pub dist: Vec<Vec<f64>>,
    pub drones: f64,
    pub speed: f64,
    pub maintenance: f64,
}

impl Data {
    fn service(&self, i: usize, j: usize) -> f64 {
        self.maintenance + 2.0 * 0.001 * self.dist[i][j] / self.speed
    }
}

/// Columns: x_j, then y_ij at nb + i·nb + j.
fn base(d: &Data, notebook: bool) -> Model {
    let (ni, nb) = (d.rate.len(), d.bases.len());
    let total: f64 = d.rate.iter().sum();
    let mut m = Model::new(false);
    for _ in 0..nb {
        m.bin(0.0);
    }
    let y = |i: usize, j: usize| nb + i * nb + j;
    for i in 0..ni {
        for j in 0..nb {
            let tau = if notebook { d.dist[i][j] / d.speed } else { 0.001 * d.dist[i][j] / d.speed };
            m.bin(d.rate[i] * tau / total);
        }
    }
    for i in 0..ni {
        for j in 0..nb {
            m.le([(y(i, j), 1.0), (j, -1.0)], 0.0);
        }
        m.eq((0..nb).map(|j| (y(i, j), 1.0)), 1.0);
    }
    m.le((0..nb).map(|j| (j, 1.0)), d.drones);
    for j in 0..nb {
        m.le((0..ni).map(|i| (y(i, j), d.rate[i] * d.service(i, j))), 1.0);
    }
    m
}

/// The notebook's linear model: V_j at nb + ni·nb + j, then ν_ij.
pub fn model(d: &Data, notebook: bool) -> Model {
    let (ni, nb) = (d.rate.len(), d.bases.len());
    let total: f64 = d.rate.iter().sum();
    let mut m = base(d, notebook);
    // At 1e-6 the McCormick rows' slack moves the optimum by 1e-6 relative (measured against the
    // exact quadratic model).
    m.feastol = Some(1e-9);
    let y = |i: usize, j: usize| nb + i * nb + j;
    let vbar = 2.0 * (0..ni).flat_map(|i| (0..nb).map(move |j| (i, j))).map(|(i, j)| d.service(i, j)).sum::<f64>();
    let v: Vec<usize> = (0..nb).map(|_| m.var(0.0, vbar, 0.0)).collect();
    let nu: Vec<Vec<usize>> =
        (0..ni).map(|i| (0..nb).map(|_| m.var(0.0, INF, 0.5 * d.rate[i] / total)).collect()).collect();
    for j in 0..nb {
        let mut r = vec![(v[j], 1.0)];
        for i in 0..ni {
            let s = d.service(i, j);
            r.extend([(y(i, j), -d.rate[i] * s * s), (nu[i][j], -d.rate[i] * s)]);
        }
        m.eq(r, 0.0);
        for (i, nui) in nu.iter().enumerate() {
            m.le([(nui[j], 1.0), (v[j], -1.0)], 0.0);
            m.le([(nui[j], 1.0), (y(i, j), -vbar)], 0.0);
            m.ge([(nui[j], 1.0), (y(i, j), -vbar), (v[j], -1.0)], -vbar);
        }
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d, false);
    let s = m.solve();
    let (ni, nb) = (d.rate.len(), d.bases.len());
    let mut text = String::new();
    let _ = writeln!(text, "mean wait + flight {:.4} min", 60.0 * s.obj);
    for j in 0..nb {
        if s.x[j] > 0.5 {
            let k = (0..ni).filter(|&i| s.x[nb + i * nb + j] > 0.5).count();
            let _ = writeln!(text, "  {}: {k} locations", d.bases[j]);
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Data, f64) {
        let p = format!("{}/../../data/drone_network.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        (serde_json::from_value(v["data"].clone()).unwrap(), v["gurobi"][0].as_f64().unwrap())
    }

    /// The notebook's objective term gives Gurobi's optimum.
    #[test]
    fn notebook_rows_give_gurobi_optimum() {
        let (d, z) = data();
        let s = model(&d, true).solve();
        assert!((s.obj - z).abs() <= 1e-6 * z, "{} against {z}", s.obj);
    }

    /// The queue written directly, V_j (1 − ρ_j) = Σ_i λ_i s_ij² y_ij with the wait term
    /// ½ Σ_i λ_i V_j y_ij, has no McCormick rows and no cap V̄; SCIP's global optimum of that
    /// nonconvex model equals the linear model's on HiGHS.
    #[test]
    fn linearization_is_exact() {
        let (d, _) = data();
        let (ni, nb) = (d.rate.len(), d.bases.len());
        let total: f64 = d.rate.iter().sum();
        let mut q = base(&d, false);
        q.feastol = Some(1e-9);
        // SCIP needs V_j bounded. A used base serves some i, whose wait term alone is
        // λ_i V_j / (2Λ); so V_j > 2Λz/λ_min costs more than z, the HiGHS optimum, and the bound
        // cuts no better solution.
        let z = model(&d, false).solve().obj;
        let lmin = d.rate.iter().copied().fold(f64::INFINITY, f64::min);
        let cap = 2.0 * total * z / lmin;
        let y = |i: usize, j: usize| nb + i * nb + j;
        for j in 0..nb {
            let v = q.var(0.0, cap, 0.0);
            let lin = [(v, 1.0)].into_iter().chain((0..ni).map(|i| (y(i, j), -d.rate[i] * d.service(i, j).powi(2))));
            let quad: Vec<(usize, usize, f64)> = (0..ni).map(|i| (v, y(i, j), -d.rate[i] * d.service(i, j))).collect();
            q.qrow(0.0, lin, quad, 0.0);
            for i in 0..ni {
                q.qobj(v, y(i, j), 0.5 * d.rate[i] / total);
            }
        }
        let s = q.solve();
        assert!((s.obj - z).abs() <= 1e-6 * z, "SCIP {} against HiGHS {z}", s.obj);
    }
}
