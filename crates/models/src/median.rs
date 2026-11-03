//! Store siting on clustered customers: a p-median model on k-means centroids. Source: Gurobi
//! modeling-examples `customer_assignment`.
//!
//! Candidate sites f, customer clusters c with weight w_c (customers in c) and distances δ_fc,
//! kept when δ_fc < θ; at most p sites. Columns: x_f site open, y_fc cluster c served by f, binary:
//!
//!   Σ_f x_f ≤ p,   y_fc ≤ x_f,   Σ_f y_fc = 1;   min Σ w_c δ_fc y_fc.
//!
//! The notebook seeds Python's `random` but draws the customers per Gaussian from numpy's
//! unseeded global generator, so its instance changes on every run and its Gurobi objective is
//! not reproducible; the importer seeds numpy with the same seed.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub sites: usize,
    pub weights: Vec<f64>,
    /// (site, cluster, δ).
    pub pairs: Vec<(usize, usize, f64)>,
    pub max_sites: f64,
}

pub fn model(d: &Data) -> Model {
    let mut m = Model::new(false);
    let x: Vec<usize> = (0..d.sites).map(|_| m.bin(0.0)).collect();
    let mut by = vec![Vec::new(); d.weights.len()];
    for &(f, c, dist) in &d.pairs {
        let y = m.bin(d.weights[c] * dist);
        m.le([(y, 1.0), (x[f], -1.0)], 0.0);
        by[c].push((y, 1.0));
    }
    m.le(x.iter().map(|&j| (j, 1.0)), d.max_sites);
    for r in by {
        m.eq(r, 1.0);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    let open: Vec<String> = (0..d.sites).filter(|&f| s.x[f] > 0.5).map(|f| f.to_string()).collect();
    let _ = writeln!(text, "weighted distance {:.4}, sites {}", s.obj, open.join(" "));
    Out::of(&m, &s, vec![s.obj], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every set of at most p sites, each cluster at its nearest open site in range: the least
    /// total is the solver's optimum.
    #[test]
    fn meets_enumeration() {
        let p = format!("{}/../../data/customer_assignment.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let d: Data = serde_json::from_value(v["data"].clone()).unwrap();
        let mut cost = vec![vec![f64::INFINITY; d.weights.len()]; d.sites];
        for &(f, c, dist) in &d.pairs {
            cost[f][c] = d.weights[c] * dist;
        }
        let mut best = f64::INFINITY;
        for s in 1..1usize << d.sites {
            if s.count_ones() as f64 > d.max_sites {
                continue;
            }
            let z: f64 = (0..d.weights.len())
                .map(|c| (0..d.sites).filter(|&f| s >> f & 1 == 1).map(|f| cost[f][c]).fold(f64::INFINITY, f64::min))
                .sum();
            best = best.min(z);
        }
        let z = model(&d).solve().obj;
        assert!((z - best).abs() <= 1e-6 * best, "{z} against {best}");
    }
}
