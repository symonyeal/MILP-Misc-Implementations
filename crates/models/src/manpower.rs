//! Manpower planning: H. P. Williams, *Model Building in Mathematical Programming* (5th ed.,
//! 2013), problem "Manpower planning". Source: Gurobi modeling-examples `manpower_planning`
//! (objective 1: fewest redundancies).
//!
//! Years t, skill levels s (ordered s1 < s2 < s3): workforce w_ts, hires h_ts ≤ H_ts, part-time
//! p_ts ≤ P (each counting as a fraction φ of a worker), redundancies r_ts, overmanning o_ts,
//! retraining or downgrading g_t,s→s' (s ≠ s'). Attrition: rookies α_s, veterans β_s, downgraded
//! workers δ.
//!
//!   w_ts = (1 − β_s) w_{t−1,s} + (1 − α_s) h_ts + Σ_{s'<s} [(1 − β_s) g_t,s'→s − g_t,s→s']
//!          + Σ_{s'>s} [(1 − δ) g_t,s'→s − g_t,s→s'] − r_ts          (w_0 = current workforce),
//!   w_ts = D_ts + o_ts + φ p_ts,   Σ_s o_ts ≤ O,   g_t,1→2 ≤ G,   g_t,1→3 = 0,   g_t,2→3 ≤ γ w_t3;
//!   min Σ r.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::{Out, grid};

#[derive(Deserialize)]
pub struct Data {
    pub skills: Vec<String>,
    pub w0: Vec<f64>,
    pub need: Vec<Vec<f64>>,
    pub hire: Vec<Vec<f64>>,
    pub rookie: Vec<f64>,
    pub veteran: Vec<f64>,
    pub demoted: f64,
    pub over: f64,
    pub part: f64,
    pub part_w: f64,
    pub train12: f64,
    pub train23: f64,
}

/// The model, and the redundancy columns r[t][s].
pub fn model(d: &Data) -> (Model, Vec<Vec<usize>>) {
    let (nt, ns) = (d.need.len(), d.skills.len());
    let mut m = Model::new(false);
    let w = grid(&mut m, nt, ns, |m, _, _| m.var(0.0, INF, 0.0));
    let h = grid(&mut m, nt, ns, |m, t, s| m.var(0.0, d.hire[t][s], 0.0));
    let p = grid(&mut m, nt, ns, |m, _, _| m.var(0.0, d.part, 0.0));
    let r = grid(&mut m, nt, ns, |m, _, _| m.var(0.0, INF, 1.0));
    let o = grid(&mut m, nt, ns, |m, _, _| m.var(0.0, INF, 0.0));
    // g[t][a][b], a ≠ b; the diagonal is unused and fixed at 0.
    let g: Vec<Vec<Vec<usize>>> =
        (0..nt).map(|_| grid(&mut m, ns, ns, |m, a, b| m.var(0.0, if a == b { 0.0 } else { INF }, 0.0))).collect();
    for t in 0..nt {
        for s in 0..ns {
            let mut row = vec![(w[t][s], 1.0), (h[t][s], -(1.0 - d.rookie[s])), (r[t][s], 1.0)];
            for u in (0..ns).filter(|&u| u != s) {
                let keep = if u < s { 1.0 - d.veteran[s] } else { 1.0 - d.demoted };
                row.extend([(g[t][u][s], -keep), (g[t][s][u], 1.0)]);
            }
            let rhs = if t == 0 {
                (1.0 - d.veteran[s]) * d.w0[s]
            } else {
                row.push((w[t - 1][s], -(1.0 - d.veteran[s])));
                0.0
            };
            m.eq(row, rhs);
            m.eq([(w[t][s], 1.0), (o[t][s], -1.0), (p[t][s], -d.part_w)], d.need[t][s]);
        }
        m.le((0..ns).map(|s| (o[t][s], 1.0)), d.over);
        m.le([(g[t][0][1], 1.0)], d.train12);
        m.eq([(g[t][0][2], 1.0)], 0.0);
        m.le([(g[t][1][2], 1.0), (w[t][2], -d.train23)], 0.0);
    }
    (m, r)
}

pub fn run(d: &Data) -> Out {
    let (m, r) = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "redundancies {:.4}", s.obj);
        for (t, row) in r.iter().enumerate() {
            let v: Vec<String> = row.iter().map(|&j| format!("{:.1}", s.x[j])).collect();
            let _ = writeln!(text, "  year {}: {}", t + 1, v.join(" "));
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
