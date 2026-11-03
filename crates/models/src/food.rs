//! Food Manufacture I and II: H. P. Williams, *Model Building in Mathematical Programming*
//! (5th ed., Wiley 2013), problems "Food manufacture 1" and "Food manufacture 2". Source: Gurobi
//! modeling-examples `food_manufacturing`.
//!
//! Month t, oil o: buy b_to at price c_to, refine u_to, hold s_to; make f_t tons sold at P.
//! Vegetable oils (o ∈ V) share refining capacity C_V, the others C_W; oil o has hardness η_o.
//!
//!   max Σ_t (P f_t − Σ_o c_to b_to − h Σ_o s_to)
//!   s_{t−1,o} + b_to = u_to + s_to  (s_{−1,o} = s0),   s_{T−1,o} = target,
//!   Σ_{o∈V} u_to ≤ C_V,   Σ_{o∉V} u_to ≤ C_W,   ηmin f_t ≤ Σ_o η_o u_to ≤ ηmax f_t,   Σ_o u_to = f_t.
//!
//! II (`recipe`) adds δ_to ∈ {0, 1}, oil o used in month t:
//!   ℓ δ_to ≤ u_to ≤ C δ_to,   Σ_o δ_to ≤ K,   δ_ta ≤ δ_tb for each implication a ⇒ b.
//! The notebook's indicators δ = 0 ⇒ u = 0 and δ = 1 ⇒ u ≥ ℓ are these rows with C the capacity
//! of the oil's class, an exact bound because u_to ≤ C_V or C_W already.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Names, Out, grid, stock};

#[derive(Deserialize)]
pub struct Data {
    pub months: Vec<String>,
    pub oils: Vec<Oil>,
    pub cost: Vec<Vec<f64>>,
    pub price: f64,
    pub s0: f64,
    pub target: f64,
    pub cap_veg: f64,
    pub cap_oil: f64,
    pub hmin: f64,
    pub hmax: f64,
    pub hold: f64,
    pub recipe: Option<Recipe>,
}

#[derive(Deserialize)]
pub struct Oil {
    pub name: String,
    pub veg: bool,
    pub hard: f64,
}

#[derive(Deserialize)]
pub struct Recipe {
    pub k: f64,
    pub min: f64,
    pub implies: Vec<(String, String)>,
}

/// Column indices: f by [t]; b, u, s and δ (empty without a recipe) by [t][o].
pub struct Ix {
    pub f: Vec<usize>,
    pub b: Vec<Vec<usize>>,
    pub u: Vec<Vec<usize>>,
    pub s: Vec<Vec<usize>>,
    pub dl: Vec<Vec<usize>>,
}

pub fn model(d: &Data) -> (Model, Ix) {
    let (nt, no) = (d.months.len(), d.oils.len());
    let cap = |o: usize| if d.oils[o].veg { d.cap_veg } else { d.cap_oil };
    let mut m = Model::new(true);
    let f: Vec<usize> = (0..nt).map(|_| m.var(0.0, INF, d.price)).collect();
    let b = grid(&mut m, nt, no, |m, t, o| m.var(0.0, INF, -d.cost[t][o]));
    let u = grid(&mut m, nt, no, |m, _, _| m.var(0.0, INF, 0.0));
    let s = grid(&mut m, nt, no, |m, _, _| m.var(0.0, INF, -d.hold));
    for o in 0..no {
        let col = |x: &Vec<Vec<usize>>| (0..nt).map(|t| x[t][o]).collect::<Vec<_>>();
        stock(&mut m, d.s0, &col(&b), &col(&u), &col(&s), d.target);
    }
    for t in 0..nt {
        let class = |veg: bool| (0..no).filter(move |&o| d.oils[o].veg == veg).map(|o| (u[t][o], 1.0));
        m.le(class(true), d.cap_veg);
        m.le(class(false), d.cap_oil);
        let eta = || (0..no).map(|o| (u[t][o], d.oils[o].hard));
        m.ge(eta().chain([(f[t], -d.hmin)]), 0.0);
        m.le(eta().chain([(f[t], -d.hmax)]), 0.0);
        m.eq((0..no).map(|o| (u[t][o], 1.0)).chain([(f[t], -1.0)]), 0.0);
    }
    let mut dl = Vec::new();
    if let Some(r) = &d.recipe {
        let names = Names::new(d.oils.iter().map(|o| o.name.as_str()));
        dl = grid(&mut m, nt, no, |m, _, _| m.bin(0.0));
        for t in 0..nt {
            for o in 0..no {
                m.ge([(u[t][o], 1.0), (dl[t][o], -r.min)], 0.0);
                m.le([(u[t][o], 1.0), (dl[t][o], -cap(o))], 0.0);
            }
            m.le((0..no).map(|o| (dl[t][o], 1.0)), r.k);
            for (a, c) in &r.implies {
                m.le([(dl[t][names.at(a)], 1.0), (dl[t][names.at(c)], -1.0)], 0.0);
            }
        }
    }
    (m, Ix { f, b, u, s, dl })
}

pub fn run(d: &Data) -> Out {
    let (m, ix) = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "profit {:.2}", s.obj);
        let head: String = d.oils.iter().map(|o| format!("{:>8}", o.name)).collect();
        for (lab, x) in [("buy", &ix.b), ("refine", &ix.u), ("store", &ix.s)] {
            let _ = writeln!(text, "{lab:6} {head}");
            for (t, row) in x.iter().enumerate() {
                let _ = writeln!(
                    text,
                    "{:6} {}",
                    d.months[t],
                    row.iter().map(|&j| format!("{:8.1}", s.x[j])).collect::<String>()
                );
            }
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
