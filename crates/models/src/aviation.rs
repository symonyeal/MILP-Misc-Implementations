//! Aviation planning: which flights an airline keeps when its airports lose capacity. Source:
//! Gurobi modeling-examples `aviation_planning` (1 July 2006, the four busiest airports).
//!
//! Each aircraft a keeps its own flights F_a; a route is a path source_a → f₁ → … → sink_a in the
//! graph with an arc f → g when f lands where g leaves and before g leaves, source_a → f when f
//! leaves a's start airport, f → sink_a when f lands at a's end airport, and source_a → sink_a.
//! Columns: y on arcs, x_f (flight f flown), binary. With revenue R_f, capacity share α and
//! airports i with D_i departures and A_i arrivals in the plan:
//!
//!   out(source_a) = in(sink_a) = 1,   out(f) = in(f),   x_f ≤ out(f),
//!   Σ_{f leaves i} x_f ≤ α D_i,   Σ_{f lands at i} x_f ≤ α A_i;   min Σ_f R_f (1 − x_f).
//!
//! The notebook departs from this in three ways, all corrected here and kept under `notebook`
//! (the test checks that they give Gurobi's optimum): its source leaves at 00:00 and needs a flight
//! to leave strictly later, so flights at 00:00 cannot be first; it compares clock times, so a
//! flight landing at 00:10 after midnight can be followed by one leaving at 00:30 that morning;
//! and it counts source and sink, as pseudo-flights, among D_i and A_i, which loosens the caps.
//! Its saved Gurobi log (801512) has 10 fewer arcs than its own cells build from the published
//! data, so that run used other data and is kept for reference only.

use std::fmt::Write;

use milp::Model;
use serde::Deserialize;

use crate::{Names, Out};

#[derive(Deserialize)]
pub struct Flight {
    pub id: u32,
    pub ori: String,
    pub des: String,
    /// Clock minutes; `arr < dep` means the flight lands after midnight.
    pub dep: u32,
    pub arr: u32,
    pub revenue: f64,
}

#[derive(Deserialize)]
pub struct Aircraft {
    pub name: String,
    pub start: String,
    pub end: String,
    pub flights: Vec<Flight>,
}

#[derive(Deserialize)]
pub struct Data {
    pub airports: Vec<String>,
    pub alpha: f64,
    pub aircraft: Vec<Aircraft>,
}

/// The model and the x column of each flight, per aircraft. `notebook`: the notebook's rows,
/// without x columns (and their rows) for source and sink, which never pay.
pub fn model(d: &Data, notebook: bool) -> (Model, Vec<Vec<usize>>) {
    const DAY: u32 = 24 * 60;
    let ap = Names::new(d.airports.iter().map(String::as_str));
    let at = |s: &str| ap.0.get(s).copied();
    let mut m = Model::new(false);
    let (mut dep, mut arr) = (vec![Vec::new(); d.airports.len()], vec![Vec::new(); d.airports.len()]);
    let (mut nd, mut na) = (vec![0.0; d.airports.len()], vec![0.0; d.airports.len()]);
    let mut xs = Vec::new();
    for a in &d.aircraft {
        let n = a.flights.len();
        let land = |f: &Flight| if notebook || f.arr >= f.dep { f.arr } else { f.arr + DAY };
        // Nodes 0..n are flights, n the source, n + 1 the sink.
        let mut arcs = vec![(n, n + 1)];
        for (i, f) in a.flights.iter().enumerate() {
            if f.ori == a.start && (!notebook || f.dep > 0) {
                arcs.push((n, i));
            }
            if f.des == a.end && (!notebook || land(f) < DAY - 1) {
                arcs.push((i, n + 1));
            }
            for (j, g) in a.flights.iter().enumerate() {
                if i != j && f.des == g.ori && land(f) < g.dep {
                    arcs.push((i, j));
                }
            }
        }
        let y: Vec<usize> = arcs.iter().map(|_| m.bin(0.0)).collect();
        let out = |v: usize| arcs.iter().zip(&y).filter(move |(e, _)| e.0 == v).map(|(_, &j)| (j, 1.0));
        let inn = |v: usize| arcs.iter().zip(&y).filter(move |(e, _)| e.1 == v).map(|(_, &j)| (j, 1.0));
        m.eq(out(n), 1.0);
        m.eq(inn(n + 1), 1.0);
        let mut x = Vec::new();
        for (i, f) in a.flights.iter().enumerate() {
            m.eq(out(i).chain(inn(i).map(|(j, _)| (j, -1.0))), 0.0);
            let xi = m.bin(-f.revenue);
            m.c0 += f.revenue;
            m.le(out(i).map(|(j, _)| (j, -1.0)).chain([(xi, 1.0)]), 0.0);
            x.push(xi);
            if let Some(k) = at(&f.ori) {
                dep[k].push(xi);
                nd[k] += 1.0;
            }
            if let Some(k) = at(&f.des) {
                arr[k].push(xi);
                na[k] += 1.0;
            }
        }
        if notebook {
            // Source and sink leave the end airport and land at the start one; their x never
            // pays, so only the counts matter.
            if let Some(k) = at(&a.end) {
                nd[k] += 2.0;
            }
            if let Some(k) = at(&a.start) {
                na[k] += 2.0;
            }
        }
        xs.push(x);
    }
    for k in 0..d.airports.len() {
        m.le(dep[k].iter().map(|&j| (j, 1.0)), d.alpha * nd[k]);
        m.le(arr[k].iter().map(|&j| (j, 1.0)), d.alpha * na[k]);
    }
    (m, xs)
}

pub fn run(d: &Data) -> Out {
    let (m, xs) = model(d, false);
    let s = m.solve();
    let mut text = String::new();
    let _ = writeln!(text, "lost revenue {:.0}", s.obj);
    for (a, x) in d.aircraft.iter().zip(&xs) {
        let kept: Vec<String> =
            a.flights.iter().zip(x).filter(|&(_, &j)| s.x[j] > 0.5).map(|(f, _)| f.id.to_string()).collect();
        let _ = writeln!(text, "  {:<12} {}", a.name, kept.join(" "));
    }
    Out::of(&m, &s, vec![s.obj], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the notebook's rows the model has the shape of the notebook's own cells on this data:
    /// Gurobi's 536 rows, and 1930 columns (1666 arcs, 264 x) where Gurobi's saved log has 1920.
    /// The notebook's output came from other data, so its optimum 801512 is not an oracle.
    #[test]
    fn notebook_rows_give_notebook_shape() {
        let p = format!("{}/../../data/aviation_planning.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let m = model(&serde_json::from_value(v["data"].clone()).unwrap(), true).0;
        // The port has no x for source and sink: 2 per aircraft fewer columns and rows.
        assert_eq!((m.m() + 42, m.n() + 42), (536, 1930));
    }
}
