//! Food aid supply chain: diet planning and multi-commodity flow (the World Food Programme's
//! Syria operation). Source: Gurobi modeling-examples `food_program`.
//!
//! Cities i, directed links (i, j) with transport cost t_ij, foods f with nutrient content v_fu,
//! nutrients u with per-person need m_u (at most 2 m_u); camps j with demand D_j people, suppliers
//! i with price p_if. Columns: rations r_fj per person, purchases s_fi ≤ S, flows x_f,ij:
//!
//!   m_u ≤ Σ_f v_fu r_fj ≤ 2 m_u                                   (camps j, nutrients u)
//!   Σ_k x_f,ki − Σ_k x_f,ik = D_i r_fi − s_fi                     (every city i, food f)
//!   min Σ p_if s_fi + Σ t_ij x_f,ij,
//!
//! with D_i = 0 off the camps and s_fi = 0 off the suppliers. The notebook writes the flow row only
//! at camps and suppliers, so a city that is neither would create or absorb food; every city of
//! this instance is one or the other, so the optimum is the same.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct City {
    pub name: String,
    /// D_i, people fed at this city (0 if it is not a camp).
    pub demand: f64,
    /// p_if, per food, if the city supplies.
    pub price: Option<Vec<f64>>,
}

#[derive(Deserialize)]
pub struct Data {
    pub cities: Vec<City>,
    pub edges: Vec<(String, String, f64)>,
    pub foods: Vec<String>,
    pub nutrients: Vec<String>,
    /// m_u; v[f][u].
    pub need: Vec<f64>,
    pub content: Vec<Vec<f64>>,
    /// S, the purchase bound.
    pub cap: f64,
}

pub fn model(d: &Data) -> Model {
    let at = Names::new(d.cities.iter().map(|c| c.name.as_str()));
    let e: Vec<(usize, usize, f64)> = d.edges.iter().map(|(a, b, t)| (at.at(a), at.at(b), *t)).collect();
    let mut m = Model::new(false);
    let mut bal: Vec<Vec<Vec<(usize, f64)>>> = vec![vec![Vec::new(); d.cities.len()]; d.foods.len()];
    for (i, c) in d.cities.iter().enumerate() {
        if c.demand > 0.0 {
            let r: Vec<usize> = (0..d.foods.len()).map(|_| m.var(0.0, INF, 0.0)).collect();
            for (u, &mu) in d.need.iter().enumerate() {
                m.row(mu, r.iter().zip(&d.content).map(|(&j, v)| (j, v[u])), 2.0 * mu);
            }
            for (f, &j) in r.iter().enumerate() {
                bal[f][i].push((j, -c.demand));
            }
        }
        for (f, &p) in c.price.iter().flatten().enumerate() {
            bal[f][i].push((m.var(0.0, d.cap, p), 1.0));
        }
    }
    for b in bal.iter_mut() {
        for &(i, j, t) in &e {
            let x = m.var(0.0, INF, t);
            b[j].push((x, 1.0));
            b[i].push((x, -1.0));
        }
    }
    for r in bal.into_iter().flatten() {
        m.eq(r, 0.0);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    let _ = writeln!(text, "cost {:.2} million", s.obj / 1e6);
    Out::of(&m, &s, vec![s.obj], text)
}
