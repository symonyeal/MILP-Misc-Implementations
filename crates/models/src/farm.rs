//! Farm planning (a five-year dairy farm): H. P. Williams, *Model Building in Mathematical
//! Programming* (5th ed., 2013), problem "Farm planning". Source: Gurobi modeling-examples
//! `farm_planning`; parameter names are the notebook's.
//!
//! Year t, land group l, cow age a = 1..A (a = 1 heifers, 2..A−1 dairy cows C, A the oldest):
//! herd c_ta, newborn heifers n_t, heifers sold f_t, sugar beet grown b_t, grain grown g_tl, grain
//! and beet bought and sold, overtime, housing outlay o_t (repaid in instalments), yearly profit
//! π_t ≥ 0 (the notebook's default lower bound).
//!
//!   housing    n_t + c_t1 + Σ_C c_ta − Σ_{τ≤t} o_τ ≤ housing_cap
//!   feed       gr_intake Σ_C c_ta ≤ grain bought − sold + Σ_l g_tl;   likewise sugar beet
//!   land       b_t / sb_yield + hf_land (n_t + c_t1) + Σ_l g_tl / yield_l + Σ_C c_ta ≤ land_cap,
//!              g_tl ≤ yield_l area_l
//!   labour     hf_labor (n_t + c_t1) + cow_labor Σ_C c + Σ_l gr_labor g_tl / yield_l
//!              + sb_labor b_t / sb_yield ≤ labor_cap + overtime_t
//!   ageing     c_t1 = (1 − hf_decay) n_{t−1},  c_t2 = (1 − hf_decay) c_{t−1,1},
//!              c_t,a+1 = (1 − cow_decay) c_{t−1,a} (a ∈ C)
//!   births     n_t + f_t = (birthrate / 2) Σ_C c_ta;   herd at the end in [min, max]_final_cows
//!   start      c_1a = initial_hf (a ≤ 2), initial_cows (a ≥ 3)
//!   profit     π_t = the year's sales − costs − installment Σ_{τ≤t} o_τ (the notebook's row)
//!   max Σ_t (π_t − installment (t + 4) o_t),  t = 1..T.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Out, grid};

#[derive(Deserialize)]
pub struct Data {
    pub years: usize,
    pub ages: usize,
    pub gr_area: Vec<f64>,
    pub gr_yield: Vec<f64>,
    pub sb_yield: f64,
    pub housing_cap: f64,
    pub gr_intake: f64,
    pub sb_intake: f64,
    pub hf_land: f64,
    pub land_cap: f64,
    pub hf_labor: f64,
    pub cow_labor: f64,
    pub gr_labor: f64,
    pub sb_labor: f64,
    pub labor_cap: f64,
    pub cow_decay: f64,
    pub hf_decay: f64,
    pub initial_hf: f64,
    pub initial_cows: f64,
    pub birthrate: f64,
    pub min_final_cows: f64,
    pub max_final_cows: f64,
    pub bl_price: f64,
    pub hf_price: f64,
    pub cow_price: f64,
    pub milk_price: f64,
    pub gr_price: f64,
    pub sb_price: f64,
    pub gr_cost: f64,
    pub sb_cost: f64,
    pub overtime_cost: f64,
    pub regular_time_cost: f64,
    pub hf_cost: f64,
    pub cow_cost: f64,
    pub gr_land_cost: f64,
    pub sb_land_cost: f64,
    pub installment: f64,
}

pub fn model(d: &Data) -> Model {
    let (nt, na, nl) = (d.years, d.ages, d.gr_area.len());
    let mut m = Model::new(true);
    let col = |m: &mut Model| (0..nt).map(|_| m.var(0.0, INF, 0.0)).collect::<Vec<_>>();
    let (sb, gb, gs, bb, bs, ot, out, hs, nb) = (
        col(&mut m),
        col(&mut m),
        col(&mut m),
        col(&mut m),
        col(&mut m),
        col(&mut m),
        col(&mut m),
        col(&mut m),
        col(&mut m),
    );
    let pi: Vec<usize> = (0..nt).map(|_| m.var(0.0, INF, 1.0)).collect();
    let gr = grid(&mut m, nt, nl, |m, _, _| m.var(0.0, INF, 0.0));
    let c = grid(&mut m, nt, na, |m, _, _| m.var(0.0, INF, 0.0));
    for (t, &o) in out.iter().enumerate() {
        m.c[o] = -d.installment * (t as f64 + 5.0);
    }
    // Dairy cows: ages 2..A−1 (0-based 1..na−1).
    let cr = &c;
    let cows = |t: usize, w: f64| (1..na - 1).map(move |a| (cr[t][a], w)).collect::<Vec<_>>();
    let paid = |t: usize, w: f64| (0..=t).map(|u| (out[u], w)).collect::<Vec<_>>();
    for t in 0..nt {
        let young = [(nb[t], 1.0), (c[t][0], 1.0)];
        m.le(young.into_iter().chain(cows(t, 1.0)).chain(paid(t, -1.0)), d.housing_cap);
        let grown = (0..nl).map(|l| (gr[t][l], -1.0));
        m.le(cows(t, d.gr_intake).into_iter().chain([(gb[t], -1.0), (gs[t], 1.0)]).chain(grown), 0.0);
        m.le(cows(t, d.sb_intake).into_iter().chain([(bb[t], -1.0), (bs[t], 1.0), (sb[t], -1.0)]), 0.0);
        for l in 0..nl {
            m.le([(gr[t][l], 1.0)], d.gr_yield[l] * d.gr_area[l]);
        }
        let land = (0..nl).map(|l| (gr[t][l], 1.0 / d.gr_yield[l]));
        let heif = [(sb[t], 1.0 / d.sb_yield), (nb[t], d.hf_land), (c[t][0], d.hf_land)];
        m.le(heif.into_iter().chain(land).chain(cows(t, 1.0)), d.land_cap);
        let lab = (0..nl).map(|l| (gr[t][l], d.gr_labor / d.gr_yield[l]));
        let heif = [(nb[t], d.hf_labor), (c[t][0], d.hf_labor), (sb[t], d.sb_labor / d.sb_yield), (ot[t], -1.0)];
        m.le(heif.into_iter().chain(cows(t, d.cow_labor)).chain(lab), d.labor_cap);
        if t > 0 {
            m.eq([(c[t][0], 1.0), (nb[t - 1], -(1.0 - d.hf_decay))], 0.0);
            m.eq([(c[t][1], 1.0), (c[t - 1][0], -(1.0 - d.hf_decay))], 0.0);
            for a in 1..na - 1 {
                m.eq([(c[t][a + 1], 1.0), (c[t - 1][a], -(1.0 - d.cow_decay))], 0.0);
            }
        }
        m.eq([(nb[t], 1.0), (hs[t], 1.0)].into_iter().chain(cows(t, -d.birthrate / 2.0)), 0.0);
        // π_t − sales + costs = −regular_time_cost.
        let herd = d.bl_price * d.birthrate / 2.0 + d.milk_price - d.cow_cost;
        let mut r = vec![
            (pi[t], 1.0),
            (hs[t], -d.hf_price),
            (c[t][na - 1], -d.cow_price),
            (gs[t], -d.gr_price),
            (bs[t], -d.sb_price),
            (gb[t], d.gr_cost),
            (bb[t], d.sb_cost),
            (ot[t], d.overtime_cost),
            (nb[t], d.hf_cost),
            (c[t][0], d.hf_cost),
            (sb[t], d.sb_land_cost / d.sb_yield),
        ];
        r.extend(cows(t, -herd));
        r.extend((0..nl).map(|l| (gr[t][l], d.gr_land_cost / d.gr_yield[l])));
        r.extend(paid(t, d.installment));
        m.eq(r, -d.regular_time_cost);
    }
    m.row(d.min_final_cows, cows(nt - 1, 1.0), d.max_final_cows);
    for a in 0..na {
        m.eq([(c[0][a], 1.0)], if a < 2 { d.initial_hf } else { d.initial_cows });
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "discounted profit {:.4}", s.obj);
    }
    Out::of(&m, &s, vec![s.obj], text)
}
