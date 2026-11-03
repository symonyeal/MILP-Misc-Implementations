//! A `Model` solved by SCIP 10 through russcip. SCIP takes a linear objective, so x'Q₀x moves into
//! an epigraph row: min c'x + z with x'Q₀x − z ≤ 0 (max: ≥ 0), z free.

use russcip::prelude::*;
use russcip::{Status as St, Variable};

use crate::{Model, Sol, Status};

fn col(s: &mut russcip::Model<russcip::ProblemCreated>, l: f64, u: f64, c: f64, int: bool) -> Variable {
    let v = var().obj(c);
    let v = if !int {
        v.cont(l..=u)
    } else {
        // Integer bounds are integral or infinite, so the casts are exact.
        match (l.is_finite(), u.is_finite()) {
            (true, true) => v.int(l as isize..=u as isize),
            (true, false) => v.int(l as isize..),
            (false, true) => v.int(..=u as isize),
            (false, false) => v.int(..),
        }
    };
    s.add(v)
}

pub fn solve(m: &Model) -> Sol {
    let s = russcip::Model::default().hide_output();
    let mut s = if m.max { s.maximize() } else { s.minimize() };
    let x: Vec<Variable> = (0..m.n()).map(|j| col(&mut s, m.l[j], m.u[j], m.c[j], m.int[j])).collect();
    let mut q: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); m.m()];
    for &(i, j, k, v) in &m.qa {
        q[i].push((j, k, v));
    }
    for i in 0..m.m() {
        let r = m.p[i] as usize..m.p[i + 1] as usize;
        let lin: Vec<&Variable> = m.j[r.clone()].iter().map(|&j| &x[j as usize]).collect();
        let mut a = m.a[r].to_vec();
        if q[i].is_empty() {
            s.add(cons().coefs(lin, a).bounds(m.lo[i], m.hi[i]));
        } else {
            let (q1, q2): (Vec<&Variable>, Vec<&Variable>) = q[i].iter().map(|&(j, k, _)| (&x[j], &x[k])).unzip();
            let mut v: Vec<f64> = q[i].iter().map(|t| t.2).collect();
            s.add_cons_quadratic(lin, &mut a, q1, q2, &mut v, m.lo[i], m.hi[i], &format!("q{i}"));
        }
    }
    if !m.qc.is_empty() {
        let z = s.add(var().obj(1.0).cont(..));
        let (q1, q2): (Vec<&Variable>, Vec<&Variable>) = m.qc.iter().map(|&(j, k, _)| (&x[j], &x[k])).unzip();
        let mut v: Vec<f64> = m.qc.iter().map(|t| t.2).collect();
        let (lo, hi) = if m.max { (0.0, f64::INFINITY) } else { (f64::NEG_INFINITY, 0.0) };
        s.add_cons_quadratic(vec![&z], &mut [-1.0], q1, q2, &mut v, lo, hi, "obj");
    }
    let r = s.solve();
    let st = match r.status() {
        St::Optimal => Status::Optimal,
        St::Infeasible => Status::Infeasible,
        St::Unbounded | St::Inforunbd => Status::Unbounded,
        St::NodeLimit
        | St::TotalNodeLimit
        | St::StallNodeLimit
        | St::TimeLimit
        | St::MemoryLimit
        | St::GapLimit
        | St::PrimalLimit
        | St::DualLimit
        | St::SolutionLimit
        | St::BestSolutionLimit
        | St::RestartLimit => Status::Limit,
        _ => Status::Error(-1),
    };
    let mut out = Sol { st, obj: f64::NAN, bound: f64::NAN, x: vec![], y: vec![] };
    if matches!(st, Status::Optimal | Status::Limit)
        && let Some(b) = r.best_sol()
    {
        out.x = x.iter().map(|v| b.val(v)).collect();
        out.obj = m.obj(&out.x);
        out.bound = r.best_bound() + m.c0;
    }
    out
}
