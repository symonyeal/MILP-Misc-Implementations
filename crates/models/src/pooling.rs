//! The standard pooling problem in its two classical bilinear forms. Source: Gurobi
//! modeling-examples `pooling` (std_pooling). P-formulation: C. A. Haverly, "Studies of the
//! behavior of recursion for the pooling problem", ACM SIGMAP Bulletin 25 (1978); Q-formulation:
//! A. Ben-Tal, G. Eiger, V. Gershovitz, "Global minimization by reducing the duality gap", Math.
//! Programming 63 (1994). Both are nonconvex; SCIP solves them to global optimality.
//!
//! Sources i (cost c_i, supply S_i, quality λ_iq), pools j (capacity C_j), targets k (price P_k,
//! demand D_k, quality bounds lo_kq ≤ · ≤ hi_kq). Flows: x_ik direct (arcs given), y_ij source to
//! pool and z_jk pool to target (all pairs).
//!
//!   P: pool quality π_jq;   Σ_i λ_iq y_ij = π_jq Σ_k z_jk,   Σ_i y_ij = Σ_k z_jk,
//!      target quality Σ_i λ_iq x_ik + Σ_j π_jq z_jk ∈ [lo_kq, hi_kq] · (Σ_i x_ik + Σ_j z_jk).
//!   Q: source shares q_ij ∈ [0, 1], Σ_i q_ij = 1, so y_ij = q_ij Σ_k z_jk and pool quality is
//!      Σ_i λ_iq q_ij; supply and target quality rows carry the products q_ij z_jk.
//!   Both: Σ_j y_ij + Σ_k x_ik ≤ S_i,   Σ_k z_jk ≤ C_j,   Σ_i x_ik + Σ_j z_jk ≥ D_k;
//!         max Σ_k P_k (inflow of k) − Σ_i c_i (outflow of i).
//!
//! π_jq gets the bounds [min_i λ_iq, max_i λ_iq]: a pool with flow mixes its sources, so this
//! changes no optimum, and spatial branching needs the bounds.

use std::collections::BTreeMap;
use std::fmt::Write;

use milp::{INF, Model, Sol};
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Data {
    pub attrs: Vec<String>,
    pub sources: Vec<Source>,
    pub targets: Vec<Target>,
    pub pools: Vec<Pool>,
    pub direct: Vec<(String, String, Option<f64>)>,
}

#[derive(Deserialize)]
pub struct Source {
    pub name: String,
    pub cost: f64,
    pub supply: f64,
    pub q: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Target {
    pub name: String,
    pub price: f64,
    pub demand: f64,
    pub lo: BTreeMap<String, f64>,
    pub hi: BTreeMap<String, f64>,
}

#[derive(Deserialize)]
pub struct Pool {
    pub name: String,
    pub cap: f64,
}

/// Column indices: x by direct arc, z by [j][k], and y by [i][j] (P) or q by [i][j] (Q).
pub struct Ix {
    pub x: Vec<usize>,
    pub z: Vec<Vec<usize>>,
    pub yq: Vec<Vec<usize>>,
}

/// The P model, or with `q` the Q model.
pub fn model(d: &Data, q: bool) -> (Model, Ix) {
    let (si, tk) =
        (Names::new(d.sources.iter().map(|s| s.name.as_str())), Names::new(d.targets.iter().map(|t| t.name.as_str())));
    let an = Names::new(d.attrs.iter().map(String::as_str));
    let (ns, np, nt, na) = (d.sources.len(), d.pools.len(), d.targets.len(), d.attrs.len());
    let arcs: Vec<(usize, usize)> = d.direct.iter().map(|(i, k, _)| (si.at(i), tk.at(k))).collect();
    let mut m = Model::new(true);
    let x: Vec<usize> = d
        .direct
        .iter()
        .zip(&arcs)
        .map(|((_, _, u), &(i, k))| m.var(0.0, u.unwrap_or(INF), d.targets[k].price - d.sources[i].cost))
        .collect();
    let z: Vec<Vec<usize>> = (0..np).map(|_| (0..nt).map(|k| m.var(0.0, INF, d.targets[k].price)).collect()).collect();
    let yq: Vec<Vec<usize>> = (0..ns)
        .map(|i| (0..np).map(|_| if q { m.var(0.0, 1.0, 0.0) } else { m.var(0.0, INF, -d.sources[i].cost) }).collect())
        .collect();
    let out = |j: usize| (0..nt).map(move |k| (j, k));
    // Source capacity, then Q's objective cost of pooled supply: −c_i Σ_j q_ij Σ_k z_jk.
    for i in 0..ns {
        let dx = arcs.iter().enumerate().filter(|(_, a)| a.0 == i).map(|(a, _)| (x[a], 1.0));
        if q {
            let qz: Vec<(usize, usize, f64)> = (0..np).flat_map(out).map(|(j, k)| (yq[i][j], z[j][k], 1.0)).collect();
            for &(a, b, _) in &qz {
                m.qobj(a, b, -d.sources[i].cost);
            }
            m.qrow(-INF, dx, qz, d.sources[i].supply);
        } else {
            m.le(dx.chain((0..np).map(|j| (yq[i][j], 1.0))), d.sources[i].supply);
        }
    }
    for (j, p) in d.pools.iter().enumerate() {
        m.le(out(j).map(|(j, k)| (z[j][k], 1.0)), p.cap);
        if q {
            m.eq((0..ns).map(|i| (yq[i][j], 1.0)), 1.0);
        } else {
            m.eq((0..ns).map(|i| (yq[i][j], 1.0)).chain(out(j).map(|(j, k)| (z[j][k], -1.0))), 0.0);
        }
    }
    let inflow = |k: usize| -> Vec<usize> {
        let d = arcs.iter().enumerate().filter(|(_, a)| a.1 == k).map(|(a, _)| x[a]);
        d.chain((0..np).map(|j| z[j][k])).collect()
    };
    for (k, t) in d.targets.iter().enumerate() {
        m.ge(inflow(k).into_iter().map(|c| (c, 1.0)), t.demand);
    }
    // P: pool quality columns and their balance rows.
    let mut pi = vec![vec![0usize; na]; np];
    if !q {
        for j in 0..np {
            for a in 0..na {
                let (lo, hi) = d.sources.iter().fold((INF, -INF), |(l, h), s| (l.min(s.q[a]), h.max(s.q[a])));
                pi[j][a] = m.var(lo, hi, 0.0);
                let y = (0..ns).map(|i| (yq[i][j], d.sources[i].q[a]));
                m.qrow(0.0, y, out(j).map(|(j, k)| (pi[j][a], z[j][k], -1.0)), 0.0);
            }
        }
    }
    // Target quality: Σ λ x + (pooled quality)·z − b·inflow, ≥ 0 for lo and ≤ 0 for hi.
    for (k, t) in d.targets.iter().enumerate() {
        for (bound, ge) in [(&t.lo, true), (&t.hi, false)] {
            for (attr, &b) in bound {
                let a = an.at(attr);
                let lin: Vec<(usize, f64)> = arcs
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.1 == k)
                    .map(|(e, &(i, _))| (x[e], d.sources[i].q[a]))
                    .chain(inflow(k).into_iter().map(|c| (c, -b)))
                    .collect();
                let quad: Vec<(usize, usize, f64)> = if q {
                    (0..ns)
                        .flat_map(|i| (0..np).map(move |j| (i, j)))
                        .map(|(i, j)| (yq[i][j], z[j][k], d.sources[i].q[a]))
                        .collect()
                } else {
                    (0..np).map(|j| (pi[j][a], z[j][k], 1.0)).collect()
                };
                if ge {
                    m.qrow(0.0, lin, quad, INF)
                } else {
                    m.qrow(-INF, lin, quad, 0.0)
                };
            }
        }
    }
    (m, Ix { x, z, yq })
}

fn report(d: &Data, ix: &Ix, s: &Sol, q: bool, text: &mut String) {
    let _ = writeln!(text, "{}-formulation: profit {:.4}", if q { "Q" } else { "P" }, s.obj);
    for (k, t) in d.targets.iter().enumerate() {
        let pooled: f64 = (0..d.pools.len()).map(|j| s.x[ix.z[j][k]]).sum();
        let direct: f64 =
            d.direct.iter().zip(&ix.x).filter(|((_, tk, _), _)| *tk == t.name).map(|(_, &c)| s.x[c]).sum();
        let _ = writeln!(text, "  {}: {:.1} pooled, {:.1} direct", t.name, pooled, direct);
    }
}

pub fn run(d: &Data) -> Out {
    let mut text = String::new();
    let runs: Vec<(Model, Sol)> = [true, false]
        .into_iter()
        .map(|q| {
            let (m, ix) = model(d, q);
            let s = m.solve();
            if !s.x.is_empty() {
                report(d, &ix, &s, q, &mut text);
            }
            (m, s)
        })
        .collect();
    Out::all(&runs, text)
}
