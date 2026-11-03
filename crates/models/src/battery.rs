//! Battery scheduling: a home battery behind the meter over one day. Source: Gurobi
//! modeling-examples `battery_scheduling`.
//!
//! Hours t with step Δ, import and export prices π_t, ρ_t, load L_t and solar G_t; battery of
//! capacity E, power limits P_c, P_d, efficiencies η_c, η_d, initial charge S_0 and end charge at
//! least S_T. Columns: charge c_t, discharge d_t, state s_t ∈ [0, E]:
//!
//!   s_t = s_{t−1} + η_c c_t Δ − d_t Δ / η_d   (s_{−1} = S_0),   s_{T−1} ≥ S_T.
//!
//! Storage only ("S", the variant the notebook runs, with P_d = P_c): max Σ (ρ_t d_t − π_t c_t) Δ.
//! Load, generation and storage ("LGS") adds grid import i_t and export e_t with
//! i_t − e_t + d_t − c_t = L_t − G_t and a degradation cost κ Σ_t (Δ (c_t + d_t) / E)^β:
//!   max Σ (ρ_t e_t − π_t i_t) Δ − κ Σ_t y_t,   y_t ≥ q_t^β,   q_t = Δ (c_t + d_t) / E.
//! For β = 3/2, y ≥ q^{3/2} (y, q ≥ 0) holds exactly when some r ≥ 0 has r² ≤ q and q² ≤ r y
//! (take r = √q), so LGS is a quadratically constrained model for SCIP. The notebook saves no
//! output, so neither variant has a Gurobi oracle; the test checks LGS against an outer
//! approximation of the convex cost by tangent cuts on HiGHS.

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    /// "S" or "LGS".
    pub variant: String,
    pub dt: f64,
    pub load: Vec<f64>,
    pub pv: Vec<f64>,
    pub import: Vec<f64>,
    pub export: Vec<f64>,
    pub e_max: f64,
    pub p_ch: f64,
    pub p_dis: f64,
    pub eta_ch: f64,
    pub eta_dis: f64,
    pub soc0: f64,
    pub soc_end: f64,
    pub kappa: f64,
    pub beta: f64,
}

/// Columns c_t, d_t, s_t at 3t, 3t+1, 3t+2; for LGS, i_t, e_t at 3T + 2t, 3T + 2t + 1, and the
/// degradation columns after. With `cuts`, the LGS cost y_t ≥ q_t^β is the tangent rows at those
/// points of q instead of the exact quadratic rows.
pub fn model(d: &Data, cuts: Option<&[f64]>) -> Model {
    let nt = d.load.len();
    let lgs = d.variant == "LGS";
    let mut m = Model::new(true);
    let p_dis = if lgs { d.p_dis } else { d.p_ch };
    for t in 0..nt {
        let (cc, cd) = if lgs { (0.0, 0.0) } else { (-d.import[t] * d.dt, d.export[t] * d.dt) };
        m.var(0.0, d.p_ch, cc);
        m.var(0.0, p_dis, cd);
        m.var(0.0, d.e_max, 0.0);
    }
    for t in 0..nt {
        let mut r = vec![(3 * t + 2, 1.0), (3 * t, -d.eta_ch * d.dt), (3 * t + 1, d.dt / d.eta_dis)];
        if t > 0 {
            r.push((3 * t - 1, -1.0));
        }
        m.eq(r, if t == 0 { d.soc0 } else { 0.0 });
    }
    m.ge([(3 * nt - 1, 1.0)], d.soc_end);
    if !lgs {
        return m;
    }
    for t in 0..nt {
        let i = m.var(0.0, INF, -d.import[t] * d.dt);
        let e = m.var(0.0, INF, d.export[t] * d.dt);
        m.eq([(i, 1.0), (e, -1.0), (3 * t + 1, 1.0), (3 * t, -1.0)], d.load[t] - d.pv[t]);
    }
    // q_t ≤ Δ (P_c + P_d) / E bounds every degradation column, as SCIP needs.
    let qmax = d.dt * (d.p_ch + p_dis) / d.e_max;
    for t in 0..nt {
        let q = m.var(0.0, qmax, 0.0);
        m.eq([(q, 1.0), (3 * t, -d.dt / d.e_max), (3 * t + 1, -d.dt / d.e_max)], 0.0);
        let y = m.var(0.0, qmax.powf(d.beta), -d.kappa);
        match cuts {
            // Tangent at q̂: y ≥ q̂^β + β q̂^{β−1} (q − q̂).
            Some(at) => {
                for &h in at {
                    let g = d.beta * h.powf(d.beta - 1.0);
                    m.ge([(y, 1.0), (q, -g)], h.powf(d.beta) - g * h);
                }
            }
            None => {
                // At SCIP's 1e-6 the rows' slack lifts the optimum 6e-5 above the tangent-cut
                // bound (measured); at 1e-9 the two agree to 1e-7.
                m.feastol = Some(1e-9);
                assert!((d.beta - 1.5).abs() < 1e-12, "the exact form needs β = 3/2");
                let r = m.var(0.0, qmax.sqrt(), 0.0);
                m.qrow(-INF, [(q, -1.0)], [(r, r, 1.0)], 0.0);
                m.qrow(-INF, [], [(q, q, 1.0), (r, y, -1.0)], 0.0);
            }
        }
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d, None);
    let s = m.solve();
    let nt = d.load.len();
    let mut text = String::new();
    let _ = writeln!(text, "{} profit {:.4}", d.variant, s.obj);
    let soc: Vec<String> = (0..nt).map(|t| format!("{:.1}", s.x[3 * t + 2])).collect();
    let _ = writeln!(text, "  charge {}", soc.join(" "));
    Out::of(&m, &s, vec![s.obj], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact LGS optimum on SCIP equals the limit of tangent-cut LPs on HiGHS, which bound it
    /// from above and tighten at each LP solution.
    #[test]
    fn lgs_meets_outer_approximation() {
        let p = format!("{}/../../data/battery_lgs.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let d: Data = serde_json::from_value(v["data"].clone()).unwrap();
        let exact = model(&d, None).solve().obj;
        let nt = d.load.len();
        let mut at: Vec<f64> = (0..=20).map(|k| k as f64 * 0.05).collect();
        let mut ub = f64::INFINITY;
        for _ in 0..60 {
            let m = model(&d, Some(&at));
            let s = m.solve();
            ub = s.obj;
            // q_t sits after c, d, s, then i, e: at 5T + 2t.
            let q: Vec<f64> = (0..nt).map(|t| s.x[5 * nt + 2 * t]).collect();
            let y: Vec<f64> = (0..nt).map(|t| s.x[5 * nt + 2 * t + 1]).collect();
            let gap: f64 = q.iter().zip(&y).map(|(q, y)| q.powf(d.beta) - y).fold(0.0, f64::max);
            if gap < 1e-10 {
                break;
            }
            at.extend(q);
        }
        assert!((ub - exact).abs() <= 1e-6 * exact.abs().max(1.0), "outer {ub}, exact {exact}");
    }
}
