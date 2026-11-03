//! Fantasy basketball lineup: eight players under a salary cap, by predicted points. Source:
//! Gurobi modeling-examples `fantasy_basketball` (part 2; part 1 fits the predictions).
//!
//! Players i with salary s_i and predicted points π_i, each eligible at a main and possibly an
//! alternative position j. Columns: y_ij, player i in position j, binary:
//!
//!   Σ_i y_ij ≥ 1 (each position j),   Σ_{j ∈ {PG, SG}} y ≥ 3,   Σ_{j ∈ {SF, PF}} y ≥ 3,
//!   Σ_j y_ij ≤ 1,   Σ y = 8,   Σ s_i y_ij ≤ S;   max Σ π_i y_ij.
//!
//! The notebook adds the position rows twice; the port once.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Player {
    pub name: String,
    pub salary: f64,
    pub points: f64,
    /// Main position, then the alternative if any.
    pub pos: Vec<String>,
}

#[derive(Deserialize)]
pub struct Data {
    pub players: Vec<Player>,
    pub cap: f64,
    pub size: f64,
}

/// The model and its columns (player, position, column).
pub fn model(d: &Data) -> (Model, Vec<(usize, String, usize)>) {
    let mut m = Model::new(true);
    let mut y = Vec::new();
    for (i, p) in d.players.iter().enumerate() {
        for j in &p.pos {
            y.push((i, j.clone(), m.bin(p.points)));
        }
    }
    let at = |set: &[&str]| -> Vec<(usize, f64)> {
        y.iter().filter(|(_, j, _)| set.contains(&j.as_str())).map(|&(_, _, c)| (c, 1.0)).collect()
    };
    let mut main: Vec<&str> = d.players.iter().map(|p| p.pos[0].as_str()).collect();
    main.sort_unstable();
    main.dedup();
    for j in main {
        m.ge(at(&[j]), 1.0);
    }
    m.ge(at(&["PG", "SG"]), 3.0);
    m.ge(at(&["SF", "PF"]), 3.0);
    for i in 0..d.players.len() {
        m.le(y.iter().filter(|t| t.0 == i).map(|t| (t.2, 1.0)), 1.0);
    }
    m.eq(y.iter().map(|t| (t.2, 1.0)), d.size);
    m.le(y.iter().map(|t| (t.2, d.players[t.0].salary)), d.cap);
    (m, y)
}

pub fn run(d: &Data) -> Out {
    let (m, y) = model(d);
    let s = m.solve();
    let mut text = String::new();
    let _ = writeln!(text, "points {:.2}", s.obj);
    for (i, j, c) in &y {
        if s.x[*c] > 0.5 {
            let p = &d.players[*i];
            let _ = writeln!(text, "  {j:<2} {:<24} {:>6.0} {:>6.2}", p.name, p.salary, p.points);
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
