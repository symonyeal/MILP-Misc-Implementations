//! Yield management: airline seat pricing over three weeks, a multistage stochastic MILP with
//! rolling re-solves. H. P. Williams, *Model Building in Mathematical Programming* (5th ed.,
//! 2013), problem "Yield management". Source: Gurobi modeling-examples `yield_management`.
//!
//! Weeks w = 0..2, demand scenarios σ with probability π_σ in each week, classes c, price options h
//! with price P_wch. A scenario path ω_w = (σ_0..σ_w); prices of week w are chosen knowing ω_{w−1},
//! sales knowing ω_w:
//!
//!   Σ_h p_w,ω_{w−1},c,h = 1,   p binary;   s_w,ω_w,c,h ≤ f_w,σ_w,c,h p_w,ω_{w−1},c,h;
//!   Σ_w Σ_h s_w,ω_w,c,h ≤ cap_c n   for every full path ω_2 and class c;   n ∈ {0..N} integer;
//!   max Σ_w Σ_{ω_w} Π π · Σ_{c,h} P_wch s_w,ω_w,c,h − F n.
//!
//! The notebook's objective has p·s in place of s; as s ≤ f p with p binary, they are equal. At
//! the start of week r the weeks before r are known: each has one scenario of probability 1 whose
//! demand is the actual demand at the option chosen then, and that option is fixed. `run` solves
//! r = 0..3 (r = 3: the realised profit). The notebook's week-3 re-solve takes week 2's chosen
//! option from the week-2 price decision of week-1 scenario j for week-2 scenario j; here it is the
//! one decision taken.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub classes: Vec<String>,
    pub prob: Vec<f64>,
    /// P[w][c][h]; f[w][σ][c][h]; actual[w][c][h].
    pub price: Vec<Vec<Vec<f64>>>,
    pub fcst: Vec<Vec<Vec<Vec<f64>>>>,
    pub actual: Vec<Vec<Vec<f64>>>,
    pub cap: Vec<f64>,
    /// F, the cost of a plane; N, the most planes.
    pub plane: f64,
    pub planes: f64,
}

/// The model at the start of week r = fixed.len(), with fixed[w][c] the option chosen for week w < r.
/// Returns it with the columns p[w][path][c][h]; n is column 0.
pub fn model(d: &Data, fixed: &[Vec<usize>]) -> (Model, Vec<Vec<Vec<Vec<usize>>>>) {
    let (nw, nc, nh) = (d.price.len(), d.cap.len(), d.price[0][0].len());
    let r = fixed.len();
    let ns = |w: usize| if w < r { 1 } else { d.prob.len() };
    let pi = |w: usize, s: usize| if w < r { 1.0 } else { d.prob[s] };
    let f = |w: usize, s: usize, c: usize, h: usize| {
        if w >= r {
            d.fcst[w][s][c][h]
        } else if h == fixed[w][c] {
            d.actual[w][c][h]
        } else {
            0.0
        }
    };
    // Paths over weeks 0..w: their number, and the index of a path's prefix (mixed radix).
    let count = |w: usize| (0..w).map(ns).product::<usize>();
    let mut m = Model::new(true);
    let n = m.int(0.0, d.planes, -d.plane);
    let mut p = Vec::new();
    for w in 0..nw {
        let pw: Vec<Vec<Vec<usize>>> = (0..count(w))
            .map(|_| {
                (0..nc)
                    .map(|c| {
                        let col: Vec<usize> = (0..nh)
                            .map(|h| {
                                let on = if h == fixed.get(w).map_or(usize::MAX, |x| x[c]) { 1.0 } else { 0.0 };
                                if w < r { m.var(on, on, 0.0) } else { m.bin(0.0) }
                            })
                            .collect();
                        m.eq(col.iter().map(|&j| (j, 1.0)), 1.0);
                        col
                    })
                    .collect()
            })
            .collect();
        p.push(pw);
    }
    // s[w][path over 0..=w][c][h], each with its row s ≤ f p and objective Π π · P.
    let mut s: Vec<Vec<Vec<Vec<usize>>>> = Vec::new();
    for w in 0..nw {
        let mut sw = Vec::new();
        for q in 0..count(w + 1) {
            let (pre, sig) = (q / ns(w), q % ns(w));
            let mut prob = pi(w, sig);
            let mut a = pre;
            for v in (0..w).rev() {
                prob *= pi(v, a % ns(v));
                a /= ns(v);
            }
            let row: Vec<Vec<usize>> = (0..nc)
                .map(|c| {
                    (0..nh)
                        .map(|h| {
                            let j = m.var(0.0, milp::INF, prob * d.price[w][c][h]);
                            m.le([(j, 1.0), (p[w][pre][c][h], -f(w, sig, c, h))], 0.0);
                            j
                        })
                        .collect()
                })
                .collect();
            sw.push(row);
        }
        s.push(sw);
    }
    for q in 0..count(nw) {
        for c in 0..nc {
            let mut row = vec![(n, -d.cap[c])];
            let mut a = q;
            for w in (0..nw).rev() {
                row.extend(s[w][a][c].iter().map(|&j| (j, 1.0)));
                a /= ns(w);
            }
            m.le(row, 0.0);
        }
    }
    (m, p)
}

pub fn run(d: &Data) -> Out {
    let nw = d.price.len();
    let mut fixed: Vec<Vec<usize>> = Vec::new();
    let mut runs = Vec::new();
    let mut text = String::new();
    for r in 0..=nw {
        let (m, p) = model(d, &fixed);
        let sol = m.solve();
        if sol.x.is_empty() {
            runs.push((m, sol));
            break;
        }
        let _ = writeln!(text, "start of week {}: profit {:.2}, planes {:.0}", r + 1, sol.obj, sol.x[0]);
        if r < nw {
            // Weeks before r are known, so week r's prices have one path, index 0.
            let pick: Vec<usize> =
                p[r][0].iter().map(|col| col.iter().position(|&j| sol.x[j] > 0.5).unwrap_or(0)).collect();
            let _ = writeln!(
                text,
                "  week {} options: {}",
                r + 1,
                d.classes.iter().zip(&pick).map(|(c, h)| format!("{c} {}", h + 1)).collect::<Vec<_>>().join(", ")
            );
            fixed.push(pick);
        }
        runs.push((m, sol));
    }
    Out::all(&runs, text)
}
