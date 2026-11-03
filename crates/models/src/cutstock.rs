//! Cutting stock by column generation (Gilmore and Gomory, Operations Research 9, 1961). Source:
//! Gurobi modeling-examples `colgen-cutting_stock`.
//!
//! Rolls of length L are cut into pieces of length l_i, d_i of each wanted. A pattern a ∈ ℤ₊^k cuts
//! a_i pieces i from one roll, Σ_i l_i a_i ≤ L; λ_a rolls are cut by pattern a.
//!
//!   master (LP over patterns P):  min Σ_{a∈P} λ_a   s.t.  Σ_{a∈P} a_i λ_a ≥ d_i,   λ ≥ 0;
//!   pricing (duals π):            max Σ_i π_i a_i   s.t.  Σ_i l_i a_i ≤ L,   a ∈ ℤ₊^k.
//!
//! The pattern priced enters while its reduced cost 1 − Σ_i π_i a_i is below −1e-9; then the LP
//! value z bounds every integer plan by ⌈z⌉. P starts with ⌊L/l_i⌋ copies of each piece alone.
//! The integer plan solves the master over the final P with λ integer, the notebook's heuristic
//! (price-and-branch): it is optimal when it reaches ⌈z⌉.
//! Master columns: λ_a by pattern, in generation order.

use std::fmt::Write;

use milp::{INF, Model, Status};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
#[allow(non_snake_case)]
pub struct Data {
    pub L: f64,
    pub pieces: Vec<Piece>,
}

#[derive(Deserialize)]
pub struct Piece {
    pub l: f64,
    pub d: f64,
}

/// The master over patterns P, λ integer when `int`.
pub fn master(d: &Data, p: &[Vec<f64>], int: bool) -> Model {
    let mut m = Model::new(false);
    for _ in p {
        if int {
            m.int(0.0, INF, 1.0)
        } else {
            m.var(0.0, INF, 1.0)
        };
    }
    for (i, pc) in d.pieces.iter().enumerate() {
        m.ge(p.iter().enumerate().map(|(k, a)| (k, a[i])), pc.d);
    }
    m
}

/// The initial patterns: ⌊L/l_i⌋ pieces i alone.
fn start(d: &Data) -> Vec<Vec<f64>> {
    let k = d.pieces.len();
    (0..k).map(|i| (0..k).map(|j| if i == j { (d.L / d.pieces[i].l).floor() } else { 0.0 }).collect()).collect()
}

pub fn model(d: &Data) -> Model {
    master(d, &start(d), false)
}

/// Column generation to the LP optimum. OUTPUT the patterns and the final master solution.
pub fn generate(d: &Data) -> (Vec<Vec<f64>>, milp::Sol) {
    let mut p = start(d);
    loop {
        let s = master(d, &p, false).solve();
        if s.st != Status::Optimal {
            return (p, s);
        }
        let mut k = Model::new(true);
        for (i, pc) in d.pieces.iter().enumerate() {
            k.int(0.0, (d.L / pc.l).floor(), s.y[i]);
        }
        k.le(d.pieces.iter().enumerate().map(|(i, pc)| (i, pc.l)), d.L);
        let r = k.solve();
        if r.st != Status::Optimal || 1.0 - r.obj >= -1e-9 {
            return (p, s);
        }
        p.push(r.x.iter().map(|x| x.round()).collect());
    }
}

pub fn run(d: &Data) -> Out {
    let (p, lp) = generate(d);
    if lp.st != Status::Optimal {
        return Out::of(&master(d, &p, false), &lp, vec![], String::new());
    }
    let m = master(d, &p, true);
    let s = m.solve();
    let mut text = String::new();
    let need: f64 = d.pieces.iter().map(|pc| pc.l * pc.d).sum::<f64>() / d.L;
    let _ = writeln!(
        text,
        "{} patterns; LP bound {:.6} (⌈LP⌉ = {}); material bound ⌈{need:.3}⌉",
        p.len(),
        lp.obj,
        (lp.obj - 1e-9).ceil()
    );
    if !s.x.is_empty() {
        let _ = writeln!(
            text,
            "rolls {} ({})",
            s.obj,
            if s.obj <= (lp.obj - 1e-9).ceil() { "optimal" } else { "gap to ⌈LP⌉ open" }
        );
        for (a, &x) in p.iter().zip(&s.x) {
            if x > 0.5 {
                let cut: Vec<String> = a
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| **n > 0.0)
                    .map(|(i, n)| format!("{n}×{}", d.pieces[i].l))
                    .collect();
                let _ = writeln!(text, "  {:3} × [{}]", x.round(), cut.join(", "));
            }
        }
    }
    Out::of(&m, &s, vec![lp.obj, s.obj], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every maximal pattern, by depth-first search over piece counts.
    fn all(d: &Data) -> Vec<Vec<f64>> {
        fn go(d: &Data, i: usize, room: f64, a: &mut Vec<f64>, out: &mut Vec<Vec<f64>>) {
            if i == d.pieces.len() {
                if d.pieces.iter().all(|pc| pc.l > room) {
                    out.push(a.clone());
                }
                return;
            }
            let mut n = 0.0;
            while n * d.pieces[i].l <= room {
                a.push(n);
                go(d, i + 1, room - n * d.pieces[i].l, a, out);
                a.pop();
                n += 1.0;
            }
        }
        let mut out = Vec::new();
        go(d, 0, d.L, &mut Vec::new(), &mut out);
        out
    }

    #[test]
    fn colgen_reaches_the_full_lp() {
        let v: serde_json::Value = serde_json::from_str(include_str!("../../../data/cutting_stock.json")).unwrap();
        let d: Data = serde_json::from_value(v["data"].clone()).unwrap();
        let (_, lp) = generate(&d);
        let full = master(&d, &all(&d), false).solve();
        assert_eq!(full.st, Status::Optimal);
        assert!((lp.obj - full.obj).abs() < 1e-6, "{} vs {}", lp.obj, full.obj);
    }
}
