//! Marketing campaign: a tactical allocation of offers to customer clusters, then an operational
//! assignment to customers. Source: Gurobi modeling-examples `marketing_campaign_optimization`.
//!
//! Tactical, clusters k with N_k customers, products j, expected profit π_kj and cost c_kj per
//! offer, budget B, minimum offers m_j, return R, penalty M on the budget increase z:
//!   Σ_j y_kj ≤ N_k,   Σ c y − z ≤ B,   Σ_k y_kj ≥ m_j,   Σ π y ≥ (1 + R) Σ c y;
//!   max Σ π y − M z,   y integer (the notebook's y is continuous and later truncated by int()).
//! Operational, customers i of cluster k with profit π_ij and cost c_ij, x binary:
//!   Σ_{i ∈ k} x_ij = y_kj,   Σ_j x_ij ≤ 1;   max Σ π x;
//! and again with the budget row Σ c x ≤ B + z.

use std::fmt::Write;

use milp::{INF, Model, Sol};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Cluster {
    pub profit: Vec<f64>,
    pub cost: Vec<f64>,
    /// Per customer: profit and cost of each product.
    pub customers: Vec<(Vec<f64>, Vec<f64>)>,
}

#[derive(Deserialize)]
pub struct Data {
    pub products: Vec<String>,
    pub clusters: Vec<Cluster>,
    pub min_offers: Vec<f64>,
    pub roi: f64,
    pub budget: f64,
    pub penalty: f64,
}

/// The tactical model; columns y_kj at k·|J| + j, then z.
pub fn model(d: &Data) -> Model {
    let nj = d.products.len();
    let mut m = Model::new(true);
    for k in &d.clusters {
        for j in 0..nj {
            m.int(0.0, INF, k.profit[j]);
        }
    }
    let z = m.var(0.0, INF, -d.penalty);
    let y = |k: usize, j: usize| k * nj + j;
    for (k, c) in d.clusters.iter().enumerate() {
        m.le((0..nj).map(|j| (y(k, j), 1.0)), c.customers.len() as f64);
    }
    let all = || d.clusters.iter().enumerate().flat_map(move |(k, c)| (0..nj).map(move |j| (k, j, c)));
    m.le(all().map(|(k, j, c)| (y(k, j), c.cost[j])).chain([(z, -1.0)]), d.budget);
    for j in 0..nj {
        m.ge((0..d.clusters.len()).map(|k| (y(k, j), 1.0)), d.min_offers[j]);
    }
    m.ge(all().map(|(k, j, c)| (y(k, j), c.profit[j] - (1.0 + d.roi) * c.cost[j])), 0.0);
    m
}

/// The operational model for offer counts y[k][j], with a budget row when `cap` is given.
fn operational(d: &Data, y: &[Vec<f64>], cap: Option<f64>) -> Model {
    let nj = d.products.len();
    let mut m = Model::new(true);
    let mut cost = Vec::new();
    for (k, c) in d.clusters.iter().enumerate() {
        let x: Vec<Vec<usize>> = c.customers.iter().map(|(p, _)| (0..nj).map(|j| m.bin(p[j])).collect()).collect();
        for j in 0..nj {
            m.eq(x.iter().map(|xi| (xi[j], 1.0)), y[k][j]);
        }
        for (xi, (_, cc)) in x.iter().zip(&c.customers) {
            m.le(xi.iter().map(|&v| (v, 1.0)), 1.0);
            cost.extend(xi.iter().zip(cc).map(|(&v, &c)| (v, c)));
        }
    }
    if let Some(b) = cap {
        m.le(cost, b);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let nj = d.products.len();
    let mt = model(d);
    let st = mt.solve();
    let mut text = String::new();
    let mut runs: Vec<(Model, Sol)> = Vec::new();
    if !st.x.is_empty() {
        let y: Vec<Vec<f64>> =
            (0..d.clusters.len()).map(|k| (0..nj).map(|j| st.x[k * nj + j].round()).collect()).collect();
        let z = st.x[d.clusters.len() * nj];
        let _ = writeln!(text, "tactical {}: offers {y:?}, budget increase {z}", st.obj);
        for cap in [None, Some(d.budget + z)] {
            let m = operational(d, &y, cap);
            let s = m.solve();
            let _ = writeln!(text, "operational{}: profit {}", if cap.is_some() { " with budget" } else { "" }, s.obj);
            runs.push((m, s));
        }
    }
    runs.insert(0, (mt, st));
    Out::all(&runs, text)
}
