//! Logical design: the fewest NOR gates realising a truth table. H. P. Williams, *Model Building
//! in Mathematical Programming* (5th ed., 2013), problem "Logical design". Source: Gurobi
//! modeling-examples `logical_design`.
//!
//! Gates g = 1..G sit in a binary tree (gate g's inputs are gates 2g and 2g + 1, when they exist,
//! and the external inputs A and B). x_g: gate used; α_g, β_g: gate g reads A, B; y_gr: g's output
//! on truth-table row r (inputs a_r, b_r). Gate 1 is used and its outputs are the table's.
//!
//!   α_g, β_g ≤ x_g;   x_2g + x_2g+1 + α_g + β_g ≤ 2;   y_cr + y_gr ≤ 1 (c a child of g);
//!   a_r α_g + y_gr ≤ 1,   b_r β_g + y_gr ≤ 1;   a_r α_g + b_r β_g + Σ_c y_cr + y_gr ≥ x_g;
//!   y_gr ≤ x_g;   min Σ_g x_g.
//!
//! The NOR rows say an input 1 forces output 0, and all inputs 0 force output 1 on a used gate.
//! Columns: x_g at g, α_g at G + g, β_g at 2G + g, y_gr at 3G + R g + r (R rows; g, r 0-based).

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub gates: usize,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub out: Vec<f64>,
}

pub fn model(d: &Data) -> Model {
    let (ng, nr) = (d.gates, d.a.len());
    let mut m = Model::new(false);
    for g in 0..ng {
        m.int(if g == 0 { 1.0 } else { 0.0 }, 1.0, 1.0);
    }
    for _ in 0..2 * ng {
        m.bin(0.0);
    }
    for g in 0..ng {
        for r in 0..nr {
            let v = if g == 0 { d.out[r] } else { -1.0 };
            if v < 0.0 {
                m.bin(0.0)
            } else {
                m.int(v, v, 0.0)
            };
        }
    }
    let (al, be, y) = (|g: usize| ng + g, |g: usize| 2 * ng + g, |g: usize, r: usize| 3 * ng + nr * g + r);
    // 0-based: gate g's children are 2g + 1 and 2g + 2.
    let kids = |g: usize| [2 * g + 1, 2 * g + 2].into_iter().filter(|&c| c < ng).collect::<Vec<_>>();
    for g in 0..ng {
        m.le([(al(g), 1.0), (g, -1.0)], 0.0);
        m.le([(be(g), 1.0), (g, -1.0)], 0.0);
        let k = kids(g);
        if !k.is_empty() {
            m.le(k.iter().map(|&c| (c, 1.0)).chain([(al(g), 1.0), (be(g), 1.0)]), 2.0);
        }
        for r in 0..nr {
            for &c in &k {
                m.le([(y(c, r), 1.0), (y(g, r), 1.0)], 1.0);
            }
            m.le([(al(g), d.a[r]), (y(g, r), 1.0)], 1.0);
            m.le([(be(g), d.b[r]), (y(g, r), 1.0)], 1.0);
            let t = [(al(g), d.a[r]), (be(g), d.b[r]), (y(g, r), 1.0), (g, -1.0)];
            m.ge(t.into_iter().chain(k.iter().map(|&c| (y(c, r), 1.0))), 0.0);
            m.ge([(g, 1.0), (y(g, r), -1.0)], 0.0);
        }
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let ng = d.gates;
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "{} NOR gates", s.obj);
        for g in (0..ng).filter(|&g| s.x[g] > 0.5) {
            let ins: Vec<&str> =
                [("A", ng + g), ("B", 2 * ng + g)].into_iter().filter(|p| s.x[p.1] > 0.5).map(|p| p.0).collect();
            let _ = writeln!(
                text,
                "  gate {} reads {}",
                g + 1,
                if ins.is_empty() { "gates only".into() } else { ins.join(", ") }
            );
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
