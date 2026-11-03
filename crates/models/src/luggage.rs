//! Lost luggage distribution: the fewest vans, then the shortest longest round, to deliver bags
//! from the depot. H. P. Williams, *Model Building in Mathematical Programming* (5th ed., 2013),
//! problem 27. Source: Gurobi modeling-examples `lost_luggage_distribution`.
//!
//! Locations i (0 the depot), vans k, travel times τ_ij. Columns: x_ijk van k drives i → j, y_ik
//! van k visits i, z_k van k used, binary; t_k ≤ T its time up to its last delivery (arcs into the
//! depot do not count, as in the notebook), s the longest:
//!
//!   y_ik ≤ z_k (i > 0),   y_0k = z_k,   Σ_k y_ik = 1 (i > 0),   Σ_i x_ijk = Σ_i x_jik = y_jk,
//!   Σ_{j > 0} τ_ij x_ijk = t_k ≤ s,   Σ_i y_i,k−1 ≥ Σ_i y_ik,
//!   Σ_{i, j ∈ S} x_ijk ≤ |S| − 1 (S a depot-free subtour met so far, every k);
//!   min Σ z_k, then min s.
//!
//! The notebook adds subtour rows lazily in Gurobi's search; here each round solves both levels
//! to optimality, adds a row per depot-free cycle of the result, and repeats. A result with no such
//! cycle is optimal: each round relaxes the full model at both levels.

use std::fmt::Write;

use milp::{Model, Sol, Status};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub points: Vec<(f64, f64)>,
    pub vans: usize,
    pub limit: f64,
}

/// Column layout: x_ijk for ordered i ≠ j, then y_ik, z_k, t_k, s.
pub struct Cols {
    pub arcs: Vec<(usize, usize)>,
    pub x: Vec<Vec<usize>>,
    pub y: Vec<Vec<usize>>,
    pub z: Vec<usize>,
    pub t: Vec<usize>,
    pub s: usize,
}

pub fn model(d: &Data) -> (Model, Cols) {
    let (n, nk) = (d.points.len(), d.vans);
    let tau = |i: usize, j: usize| (d.points[i].0 - d.points[j].0).hypot(d.points[i].1 - d.points[j].1);
    let arcs: Vec<(usize, usize)> = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).collect();
    let mut m = Model::new(false);
    let x: Vec<Vec<usize>> = arcs.iter().map(|_| (0..nk).map(|_| m.bin(0.0)).collect()).collect();
    let y: Vec<Vec<usize>> = (0..n).map(|_| (0..nk).map(|_| m.bin(0.0)).collect()).collect();
    let z: Vec<usize> = (0..nk).map(|_| m.bin(0.0)).collect();
    let t: Vec<usize> = (0..nk).map(|_| m.var(0.0, d.limit, 0.0)).collect();
    let s = m.var(0.0, milp::INF, 0.0);
    for k in 0..nk {
        for i in 1..n {
            m.le([(y[i][k], 1.0), (z[k], -1.0)], 0.0);
        }
        m.eq([(y[0][k], 1.0), (z[k], -1.0)], 0.0);
        for v in 0..n {
            let into = arcs.iter().enumerate().filter(|(_, a)| a.1 == v).map(|(a, _)| (x[a][k], 1.0));
            m.eq(into.chain([(y[v][k], -1.0)]), 0.0);
            let out = arcs.iter().enumerate().filter(|(_, a)| a.0 == v).map(|(a, _)| (x[a][k], 1.0));
            m.eq(out.chain([(y[v][k], -1.0)]), 0.0);
        }
        let time = arcs.iter().enumerate().filter(|(_, a)| a.1 > 0).map(|(a, &(i, j))| (x[a][k], tau(i, j)));
        m.eq(time.chain([(t[k], -1.0)]), 0.0);
        m.le([(t[k], 1.0), (s, -1.0)], 0.0);
        if k > 0 {
            m.ge((0..n).flat_map(|i| [(y[i][k - 1], 1.0), (y[i][k], -1.0)]), 0.0);
        }
    }
    for row in y.iter().skip(1) {
        m.eq(row.iter().map(|&c| (c, 1.0)), 1.0);
    }
    (m, Cols { arcs, x, y, z, t, s })
}

/// The depot-free cycles of a solution's arcs (each location has one successor).
fn subtours(n: usize, c: &Cols, s: &Sol) -> Vec<Vec<usize>> {
    let mut next = vec![usize::MAX; n];
    let mut leaves = vec![Vec::new(); n];
    for (a, &(i, j)) in c.arcs.iter().enumerate() {
        if c.x[a].iter().any(|&col| s.x[col] > 0.5) {
            next[i] = j;
            leaves[i].push(j);
        }
    }
    let mut seen = vec![false; n];
    for start in &leaves[0] {
        let mut v = *start;
        while v != 0 && !seen[v] {
            seen[v] = true;
            v = next[v];
        }
    }
    let mut out = Vec::new();
    for v0 in 1..n {
        if seen[v0] {
            continue;
        }
        let mut cyc = Vec::new();
        let mut v = v0;
        while !seen[v] {
            seen[v] = true;
            cyc.push(v);
            v = next[v];
        }
        out.push(cyc);
    }
    out
}

/// Both levels to optimality with subtour rows added until none is violated; the final model and
/// its two solutions.
pub fn solve(d: &Data) -> (Model, Cols, Vec<Sol>, usize) {
    let (mut m, c) = model(d);
    let n = d.points.len();
    let mut c1 = vec![0.0; m.n()];
    c.z.iter().for_each(|&j| c1[j] = 1.0);
    let mut c2 = vec![0.0; m.n()];
    c2[c.s] = 1.0;
    let mut rounds = 0;
    loop {
        rounds += 1;
        let r = m.lex(&[(&c1, 0.0), (&c2, 0.0)]);
        if r.iter().any(|s| s.st != Status::Optimal) || r.len() < 2 {
            return (m, c, r, rounds);
        }
        let cs = subtours(n, &c, &r[1]);
        if cs.is_empty() {
            return (m, c, r, rounds);
        }
        for cyc in &cs {
            for k in 0..d.vans {
                let inside = |v: usize| cyc.contains(&v);
                let row = c
                    .arcs
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| inside(a.0) && inside(a.1))
                    .map(|(a, _)| (c.x[a][k], 1.0));
                m.le(row, cyc.len() as f64 - 1.0);
            }
        }
    }
}

pub fn run(d: &Data) -> Out {
    let (m, c, r, rounds) = solve(d);
    let mut text = String::new();
    let Some(s) = r.last().filter(|_| r.len() == 2) else {
        return Out::of(&m, &r[0], vec![], text);
    };
    let vans = c.z.iter().filter(|&&j| s.x[j] > 0.5).count();
    let _ = writeln!(text, "{vans} vans, longest round {:.4} after {rounds} rounds", s.x[c.s]);
    for k in 0..d.vans {
        if s.x[c.z[k]] < 0.5 {
            continue;
        }
        let mut route = vec![0];
        let mut v = 0;
        while let Some(j) =
            c.arcs.iter().enumerate().find(|&(a, &(i, _))| i == v && s.x[c.x[a][k]] > 0.5).map(|(_, &(_, j))| j)
        {
            if j == 0 {
                break;
            }
            route.push(j);
            v = j;
        }
        let stops: Vec<String> = route.iter().map(usize::to_string).collect();
        let _ = writeln!(text, "  van {k}: {}, {:.2} min", stops.join(" "), s.x[c.t[k]]);
    }
    Out::of(&m, s, vec![r[0].obj, s.x[c.s]], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P(S): the shortest path from the depot through every customer of S, by Held and Karp's
    /// dynamic program (J. SIAM 10, 1962) over the customers 1..n, O(2ⁿ n²).
    fn paths(d: &Data) -> Vec<f64> {
        let n = d.points.len() - 1;
        let tau = |i: usize, j: usize| (d.points[i].0 - d.points[j].0).hypot(d.points[i].1 - d.points[j].1);
        let mut g = vec![f64::INFINITY; (1 << n) * n];
        for k in 0..n {
            g[(1 << k) * n + k] = tau(0, k + 1);
        }
        let mut p = vec![f64::INFINITY; 1 << n];
        p[0] = 0.0;
        for s in 1..1usize << n {
            for k in 0..n {
                let gk = g[s * n + k];
                if s >> k & 1 == 0 || gk.is_infinite() {
                    continue;
                }
                p[s] = p[s].min(gk);
                for j in 0..n {
                    if s >> j & 1 == 0 {
                        let e = &mut g[(s | 1 << j) * n + j];
                        *e = e.min(gk + tau(k + 1, j + 1));
                    }
                }
            }
        }
        p
    }

    /// One van cannot deliver within the limit, and the best split between two vans has the
    /// longest round the row-generation solve found.
    #[test]
    fn meets_exhaustive_split() {
        let p = format!("{}/../../data/lost_luggage.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let d: Data = serde_json::from_value(v["data"].clone()).unwrap();
        let pp = paths(&d);
        let all = pp.len() - 1;
        assert!(pp[all] > d.limit, "one van suffices: {}", pp[all]);
        let best = (1..all).map(|s| pp[s].max(pp[all ^ s])).fold(f64::INFINITY, f64::min);
        assert!(best <= d.limit);
        let (_, c, r, _) = solve(&d);
        let z = r[1].x[c.s];
        assert!(
            (r[0].obj - 2.0).abs() < 1e-6 && (z - best).abs() <= 1e-6 * best,
            "{} vans, {z} against {best}",
            r[0].obj
        );
    }
}
