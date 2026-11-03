//! Milk collection: one tanker, two day types; every-day farms are visited daily, the others on
//! one of the two. H. P. Williams, *Model Building in Mathematical Programming* (5th ed., 2013),
//! problem 23. Source: Gurobi modeling-examples `milk_collection`.
//!
//! Farms i (0 the depot) with yield q_i, E the every-day farms (depot included), O the others,
//! day types k ∈ {1, 2}, tanker capacity C, distances d_ij. Columns: x_ijk edge {i, j} on day k,
//! y_ik farm i ∈ O visited on day k, binary:
//!
//!   Σ_j x_ijk = 2 (i ∈ E),   Σ_j x_ijk = 2 y_ik (i ∈ O),   Σ_{i ∈ O} q_i y_ik ≤ C − Σ_{i ∈ E} q_i,
//!   Σ_k y_ik = 1,   y_{11,1} = 1 (the day types are interchangeable),
//!   Σ_{i<j ∈ S} x_ijk ≤ |S| − 1 (S a subtour met so far on day k);   min Σ d_ij x_ijk.
//!
//! The notebook adds subtour rows lazily in Gurobi's search; here each round solves to optimality,
//! adds a row per subtour of each day, and repeats until each day is one tour.

use std::fmt::Write;

use milp::{Model, Sol, Status};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub at: Vec<(f64, f64)>,
    pub collect: Vec<f64>,
    /// Farms 0..every are visited every day.
    pub every: usize,
    pub cap: f64,
    /// The farm fixed to day type 1.
    pub fix: usize,
}

const DAYS: usize = 2;

/// The edges {i, j}, i < j; x[e][k]; y[i − every][k].
pub struct Cols {
    pub e: Vec<(usize, usize)>,
    pub x: Vec<[usize; DAYS]>,
    pub y: Vec<[usize; DAYS]>,
}

pub fn dist(d: &Data, i: usize, j: usize) -> f64 {
    (d.at[i].0 - d.at[j].0).hypot(d.at[i].1 - d.at[j].1)
}

pub fn model(d: &Data) -> (Model, Cols) {
    let n = d.at.len();
    let e: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
    let mut m = Model::new(false);
    let x: Vec<[usize; DAYS]> = e.iter().map(|&(i, j)| [(); DAYS].map(|_| m.bin(dist(d, i, j)))).collect();
    let y: Vec<[usize; DAYS]> = (d.every..n).map(|_| [(); DAYS].map(|_| m.bin(0.0))).collect();
    let base: f64 = d.collect[..d.every].iter().sum();
    for k in 0..DAYS {
        for v in 0..n {
            let deg = e.iter().zip(&x).filter(|(a, _)| a.0 == v || a.1 == v).map(|(_, c)| (c[k], 1.0));
            if v < d.every {
                m.eq(deg, 2.0);
            } else {
                m.eq(deg.chain([(y[v - d.every][k], -2.0)]), 0.0);
            }
        }
        m.le((d.every..n).map(|i| (y[i - d.every][k], d.collect[i])), d.cap - base);
    }
    for c in &y {
        m.eq(c.iter().map(|&j| (j, 1.0)), 1.0);
    }
    m.l[y[d.fix - d.every][0]] = 1.0;
    (m, Cols { e, x, y })
}

/// The cycles of day k's edges.
fn cycles(n: usize, c: &Cols, s: &Sol, k: usize) -> Vec<Vec<usize>> {
    let sel: Vec<(usize, usize)> = c.e.iter().zip(&c.x).filter(|(_, x)| s.x[x[k]] > 0.5).map(|(&a, _)| a).collect();
    let mut on = vec![false; n];
    sel.iter().for_each(|&(i, j)| (on[i], on[j]) = (true, true));
    crate::tsp::cycles(n, &sel).into_iter().filter(|cy| on[cy[0]]).collect()
}

pub fn solve(d: &Data) -> (Model, Cols, Sol, usize) {
    let n = d.at.len();
    let (mut m, c) = model(d);
    let mut rounds = 0;
    loop {
        rounds += 1;
        let s = m.solve();
        if s.st != Status::Optimal {
            return (m, c, s, rounds);
        }
        let mut cut = false;
        for k in 0..DAYS {
            let cs = cycles(n, &c, &s, k);
            if cs.len() < 2 {
                continue;
            }
            cut = true;
            for cy in &cs {
                let row =
                    c.e.iter()
                        .zip(&c.x)
                        .filter(|(a, _)| cy.contains(&a.0) && cy.contains(&a.1))
                        .map(|(_, x)| (x[k], 1.0));
                m.le(row, cy.len() as f64 - 1.0);
            }
        }
        if !cut {
            return (m, c, s, rounds);
        }
    }
}

pub fn run(d: &Data) -> Out {
    let n = d.at.len();
    let (m, c, s, rounds) = solve(d);
    let mut text = String::new();
    if s.st != Status::Optimal {
        return Out::of(&m, &s, vec![], text);
    }
    let _ = writeln!(text, "distance {:.4} (x 10 miles) after {rounds} rounds", s.obj);
    for k in 0..DAYS {
        let tour: Vec<String> = cycles(n, &c, &s, k)[0].iter().map(usize::to_string).collect();
        let _ = writeln!(text, "  day {}: {}", k + 1, tour.join(" "));
    }
    Out::of(&m, &s, vec![s.obj], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The optimum by exhaustion: T(S), the shortest tour of the depot and the farms in S, for
    /// every S by Held and Karp's dynamic program (J. SIAM 10, 1962) over farms 1..n, then the best
    /// split of the other-day farms between the two days within capacity, farm `fix` on day 1.
    #[test]
    fn meets_exhaustive_split() {
        let p = format!("{}/../../data/milk_collection.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let d: Data = serde_json::from_value(v["data"].clone()).unwrap();
        let n = d.at.len() - 1;
        // g[S·n + k]: depot → all of S, ending at farm k + 1 ∈ S.
        let mut g = vec![f64::INFINITY; (1 << n) * n];
        for k in 0..n {
            g[(1 << k) * n + k] = dist(&d, 0, k + 1);
        }
        for s in 1..1usize << n {
            for k in 0..n {
                let gk = g[s * n + k];
                if gk.is_infinite() {
                    continue;
                }
                for j in 0..n {
                    if s >> j & 1 == 0 {
                        let e = &mut g[(s | 1 << j) * n + j];
                        *e = e.min(gk + dist(&d, k + 1, j + 1));
                    }
                }
            }
        }
        let tour = |s: usize| {
            (0..n).filter(|&k| s >> k & 1 == 1).map(|k| g[s * n + k] + dist(&d, k + 1, 0)).fold(f64::INFINITY, f64::min)
        };
        let every = (1 << (d.every - 1)) - 1;
        let others: Vec<usize> = (d.every..=n).collect();
        let room = d.cap - d.collect[..d.every].iter().sum::<f64>();
        let mut best = f64::INFINITY;
        for mask in 0..1usize << others.len() {
            let (mut a, mut b, mut qa, mut qb) = (every, every, 0.0, 0.0);
            for (t, &i) in others.iter().enumerate() {
                if mask >> t & 1 == 1 {
                    a |= 1 << (i - 1);
                    qa += d.collect[i];
                } else {
                    b |= 1 << (i - 1);
                    qb += d.collect[i];
                }
            }
            if a >> (d.fix - 1) & 1 == 1 && qa <= room && qb <= room {
                best = best.min(tour(a) + tour(b));
            }
        }
        let z = solve(&d).2.obj;
        assert!((z - best).abs() <= 1e-6 * best, "{z} against {best}");
    }
}
