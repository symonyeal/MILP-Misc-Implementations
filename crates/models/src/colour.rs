//! 3D tic-tac-toe as hypergraph colouring. Source: Gurobi modeling-examples `3d_tic_tac_toe`.
//!
//! Fill the n×n×n cube with k crosses and n³ − k noughts so that the fewest lines hold a single
//! symbol. The lines are the hyperedges of a hypergraph on the cells, so this is the minimum
//! number of monochromatic hyperedges over 2-colourings with one colour class of size k.
//!
//!   min Σ_e z_e   s.t.  Σ_v x_v = k,   Σ_{v ∈ e} x_v ≥ 1 − z_e,   Σ_{v ∈ e} x_v ≤ |e| − 1 + z_e,
//!                       x, z ∈ {0, 1}.
//!
//! The notebook states the two line rows as indicator constraints z_e = 0 ⇒ 1 ≤ Σ_{v∈e} x_v ≤ n − 1;
//! since 0 ≤ Σ x_v ≤ n, the rows above (big-M with M = 1) are the same set.
//! Columns: x_v at v = (i n + j) n + l, then z_e at n³ + e.

use std::fmt::Write;

use hyper::Hg;
use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub size: usize,
    pub x: usize,
}

/// The ((n + 2)³ − n³)/2 lines of the n×n×n cube. Along a line each coordinate is constant c,
/// rising t or falling n − 1 − t (t = 0..n), not all constant; of each line and its reverse, the one
/// whose first moving coordinate rises is kept.
pub fn lines(n: usize) -> Hg {
    // Coordinate mode: 0..n constant, n rising, n + 1 falling.
    let mut e = Vec::new();
    for a in 0..n + 2 {
        for b in 0..n + 2 {
            for c in 0..n + 2 {
                let md = [a, b, c];
                match md.iter().find(|&&q| q >= n) {
                    Some(&q) if q == n => {}
                    _ => continue,
                }
                let at = |q: usize, t: usize| {
                    if q < n {
                        q
                    } else if q == n {
                        t
                    } else {
                        n - 1 - t
                    }
                };
                e.push((0..n).map(|t| (at(a, t) * n + at(b, t)) * n + at(c, t)).collect());
            }
        }
    }
    Hg::new(n * n * n, e)
}

pub fn model(d: &Data) -> Model {
    let g = lines(d.size);
    let mut m = Model::new(false);
    for _ in 0..g.n {
        m.bin(0.0);
    }
    for _ in 0..g.e.len() {
        m.bin(1.0);
    }
    m.eq((0..g.n).map(|v| (v, 1.0)), d.x as f64);
    for (k, e) in g.e.iter().enumerate() {
        let z = g.n + k;
        let t = || e.iter().map(|&v| (v, 1.0));
        m.ge(t().chain([(z, 1.0)]), 1.0);
        m.le(t().chain([(z, -1.0)]), e.len() as f64 - 1.0);
    }
    m
}

pub fn run(d: &Data) -> Out {
    let m = model(d);
    let s = m.solve();
    let n = d.size;
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "{} lines, {:.0} of one symbol; layers (X cross, O nought):", lines(n).e.len(), s.obj);
        for i in 0..n {
            for j in 0..n {
                let r: String = (0..n).map(|l| if s.x[(i * n + j) * n + l] > 0.5 { 'X' } else { 'O' }).collect();
                let _ = writeln!(text, "  {r}");
            }
            text.push('\n');
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_counts() {
        // 3×3×3: the notebook's 49 lines; 4×4×4: the 76 lines of Qubic.
        assert_eq!(lines(3).e.len(), 49);
        assert_eq!(lines(4).e.len(), 76);
        let g = lines(3);
        let mut e = g.e.clone();
        e.sort();
        e.dedup();
        assert_eq!(e.len(), 49);
        assert!(g.e.iter().all(|l| l.len() == 3));
    }
}
