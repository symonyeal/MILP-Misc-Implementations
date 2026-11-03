//! Symmetric travelling salesman: the Dantzig, Fulkerson and Johnson formulation (J. Operations
//! Research Society of America 2, 1954), its subtour elimination rows generated on demand. Source: Gurobi
//! modeling-examples `traveling_salesman`.
//!
//!   min Σ_{i<j} d_ij x_ij   s.t.  Σ_j x_ij = 2 (each i),   Σ_{i<j; i,j ∈ S} x_ij ≤ |S| − 1 (S ∈ 𝒮),
//!   x binary,
//!
//! 𝒮 the subtours met so far. The notebook adds the rows lazily inside Gurobi's branch and bound;
//! HiGHS has no lazy-row callback, so each round solves the MIP to optimality, adds a row for every
//! cycle of a solution that is not one tour, and solves again. The first single-tour solution is
//! optimal, since every round relaxes the full formulation. d_ij is Euclidean in (latitude,
//! longitude) degrees, as in the notebook. Column of edge {i, j}, i < j: its rank in that order.

use std::fmt::Write;

use milp::{Model, Status};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub cities: Vec<City>,
}

#[derive(Deserialize)]
pub struct City {
    pub name: String,
    pub at: (f64, f64),
}

/// The degree model, and the edges {i, j} by column.
pub fn model(d: &Data) -> (Model, Vec<(usize, usize)>) {
    let n = d.cities.len();
    let e: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
    let mut m = Model::new(false);
    for &(i, j) in &e {
        let (a, b) = (d.cities[i].at, d.cities[j].at);
        m.bin((a.0 - b.0).hypot(a.1 - b.1));
    }
    for v in 0..n {
        m.eq(e.iter().enumerate().filter(|(_, x)| x.0 == v || x.1 == v).map(|(k, _)| (k, 1.0)), 2.0);
    }
    (m, e)
}

/// The cycles of a 2-regular edge set on n vertices, each in visiting order.
pub fn cycles(n: usize, sel: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut nb = vec![Vec::new(); n];
    for &(i, j) in sel {
        nb[i].push(j);
        nb[j].push(i);
    }
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for s in 0..n {
        if seen[s] {
            continue;
        }
        let (mut c, mut prev, mut v) = (vec![], usize::MAX, s);
        while !seen[v] {
            seen[v] = true;
            c.push(v);
            let next = nb[v].iter().copied().find(|&w| w != prev && !seen[w]);
            prev = v;
            match next {
                Some(w) => v = w,
                None => break,
            }
        }
        out.push(c);
    }
    out
}

pub fn run(d: &Data) -> Out {
    let n = d.cities.len();
    let (mut m, e) = model(d);
    let mut rounds = 0;
    loop {
        rounds += 1;
        let s = m.solve();
        if s.st != Status::Optimal {
            return Out::of(&m, &s, vec![], String::new());
        }
        let sel: Vec<(usize, usize)> = e.iter().zip(&s.x).filter(|(_, x)| **x > 0.5).map(|(&p, _)| p).collect();
        let cs = cycles(n, &sel);
        if cs.len() == 1 {
            let mut text = String::new();
            let _ = writeln!(text, "tour length {:.6} after {rounds} rounds, {} subtour rows", s.obj, m.m() - n);
            let names: Vec<&str> = cs[0].iter().map(|&v| d.cities[v].name.as_str()).collect();
            let _ = writeln!(text, "{}", names.join(" - "));
            return Out::of(&m, &s, vec![s.obj], text);
        }
        for c in &cs {
            let inside = |v: usize| c.contains(&v);
            m.le(
                e.iter().enumerate().filter(|(_, x)| inside(x.0) && inside(x.1)).map(|(k, _)| (k, 1.0)),
                c.len() as f64 - 1.0,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Held and Karp's dynamic program (J. SIAM 10, 1962): the optimal tour length, O(2ⁿ n²).
    fn held_karp(w: &[Vec<f64>]) -> f64 {
        let n = w.len();
        let mut g = vec![vec![f64::INFINITY; n]; 1 << n];
        g[1][0] = 0.0;
        for s in 1..1usize << n {
            if s & 1 == 0 {
                continue;
            }
            for k in 0..n {
                if s >> k & 1 == 0 || g[s][k].is_infinite() {
                    continue;
                }
                for j in 0..n {
                    if s >> j & 1 == 0 {
                        let t = s | 1 << j;
                        g[t][j] = g[t][j].min(g[s][k] + w[k][j]);
                    }
                }
            }
        }
        (1..n).map(|k| g[(1 << n) - 1][k] + w[k][0]).fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn matches_held_karp() {
        // Two clusters of five points force subtours in the first round.
        let mut z = 12345u64;
        let mut r = || {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (z >> 11) as f64 / (1u64 << 53) as f64
        };
        let cities: Vec<City> =
            (0..10).map(|k| City { name: k.to_string(), at: (r() + if k < 5 { 0.0 } else { 10.0 }, r()) }).collect();
        let w: Vec<Vec<f64>> =
            cities.iter().map(|a| cities.iter().map(|b| (a.at.0 - b.at.0).hypot(a.at.1 - b.at.1)).collect()).collect();
        let o = run(&Data { cities });
        assert_eq!(o.st, Status::Optimal);
        assert!((o.obj[0] - held_karp(&w)).abs() < 1e-6);
        assert!(o.viol < 1e-9);
    }
}
