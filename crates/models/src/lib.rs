//! The cases. Each module states its problem, source and formulation, and exposes `model`, the
//! first model it solves (its constraint matrix is what `mmi hgr` exports), and `run`, which solves
//! the case to its end, replays the final solution against its model and reports.
//!
//! A case file (data/*.json) is {"case": <module>, "source", "oracle": [objectives], "data": {...}};
//! the oracle lists the objectives Gurobi recorded in the source notebook, in solve order.

use std::collections::HashMap;
use std::fmt::Write;

use milp::{Model, Sol, Status};
use serde::de::DeserializeOwned;
use serde_json::Value;

pub mod agriprice;
pub mod avocado;
pub mod colour;
pub mod cover;
pub mod cutstock;
pub mod economic;
pub mod facility;
pub mod factory;
pub mod fit;
pub mod food;
pub mod manpower;
pub mod network;
pub mod opencast;
pub mod playlist;
pub mod pooling;
pub mod portfolio;
pub mod subset;
pub mod tsp;
pub mod workforce;

/// What a case returns: the final solve's status, the objective of each solve the oracle covers,
/// the final solution's largest violation of its model (`Model::viol`) and a plain-text report.
#[derive(Clone, Debug)]
pub struct Out {
    pub st: Status,
    pub obj: Vec<f64>,
    pub viol: f64,
    pub text: String,
}

impl Out {
    /// The outcome of solution s of model m, whose objective is the last of `obj`.
    fn of(m: &Model, s: &Sol, obj: Vec<f64>, text: String) -> Out {
        let viol = if s.x.is_empty() { f64::NAN } else { m.viol(&s.x) };
        let mut text = text;
        if s.st != Status::Optimal {
            let _ = writeln!(text, "status {:?}", s.st);
        }
        Out { st: s.st, obj, viol, text }
    }

    /// The outcome of a sequence of solves (model, solution), one objective each: the first
    /// non-optimal status, else optimal, and the largest violation of any solution of its model.
    fn all(runs: &[(Model, Sol)], text: String) -> Out {
        let outs: Vec<Out> = runs.iter().map(|(m, s)| Out::of(m, s, vec![s.obj], String::new())).collect();
        let st = outs.iter().map(|o| o.st).find(|&st| st != Status::Optimal).unwrap_or(Status::Optimal);
        let viol = outs.iter().map(|o| o.viol).fold(0.0, |a, b| if b > a || b.is_nan() { b } else { a });
        let mut text = text;
        for o in &outs {
            text.push_str(&o.text);
        }
        Out { st, obj: runs.iter().map(|(_, s)| s.obj).collect(), viol, text }
    }

    /// The acceptance rule: optimal, violation ≤ 1e-6, and every oracle objective met.
    pub fn pass(&self, o: &Oracle) -> bool {
        let hit = o.v.len() <= self.obj.len() && o.v.iter().zip(&self.obj).all(|(&(a, b), &z)| o.hit(a, b, z));
        self.st == Status::Optimal && self.viol <= 1e-6 && hit
    }
}

/// The Gurobi objectives a notebook recorded, in solve order. Each is an interval [a, b]: an exact
/// optimum has a = b; a MIP or nonconvex solve that stopped within its gap gives its incumbent and
/// best bound. An objective z meets [a, b] when a − t ≤ z ≤ b + t with t = max(rtol |a|, 1e-9);
/// rtol is 1e-6 unless the case file states a wider one (with a note saying why).
#[derive(Clone, Debug, Default)]
pub struct Oracle {
    pub v: Vec<(f64, f64)>,
    pub rtol: f64,
}

impl Oracle {
    fn hit(&self, a: f64, b: f64, z: f64) -> bool {
        let t = (self.rtol * a.abs()).max(1e-9);
        a - t <= z && z <= b + t
    }
}

/// A case file: (case, data, oracle). An oracle entry is a number or a pair [incumbent, bound].
pub fn read(path: &std::path::Path) -> Result<(String, Value, Oracle), String> {
    let at = |e: String| format!("{}: {e}", path.display());
    let s = std::fs::read_to_string(path).map_err(|e| at(e.to_string()))?;
    let mut v: Value = serde_json::from_str(&s).map_err(|e| at(e.to_string()))?;
    let case = v["case"].as_str().ok_or_else(|| at("no case".into()))?.to_string();
    let mut o = Oracle { v: Vec::new(), rtol: v["rtol"].as_f64().unwrap_or(1e-6) };
    for e in v["oracle"].as_array().into_iter().flatten() {
        let ab = match e {
            Value::Array(p) if p.len() == 2 => p[0].as_f64().zip(p[1].as_f64()),
            _ => e.as_f64().map(|a| (a, a)),
        };
        let (a, b) = ab.ok_or_else(|| at(format!("bad oracle entry {e}")))?;
        o.v.push((a.min(b), a.max(b)));
    }
    Ok((case, v["data"].take(), o))
}

fn de<D: DeserializeOwned>(v: Value) -> Result<D, String> {
    serde_json::from_value(v).map_err(|e| e.to_string())
}

/// `run` of the named case on its data.
pub fn run(case: &str, v: Value) -> Result<Out, String> {
    Ok(match case {
        "agriprice" => agriprice::run(&de(v)?),
        "avocado" => avocado::run(&de(v)?),
        "colour" => colour::run(&de(v)?),
        "cover" => cover::run(&de(v)?),
        "cutstock" => cutstock::run(&de(v)?),
        "economic" => economic::run(&de(v)?),
        "facility" => facility::run(&de(v)?),
        "factory" => factory::run(&de(v)?),
        "fit" => fit::run(&de(v)?),
        "food" => food::run(&de(v)?),
        "manpower" => manpower::run(&de(v)?),
        "network" => network::run(&de(v)?),
        "opencast" => opencast::run(&de(v)?),
        "playlist" => playlist::run(&de(v)?),
        "pooling" => pooling::run(&de(v)?),
        "portfolio" => portfolio::run(&de(v)?),
        "subset" => subset::run(&de(v)?),
        "tsp" => tsp::run(&de(v)?),
        "workforce" => workforce::run(&de(v)?),
        _ => return Err(format!("unknown case {case}")),
    })
}

/// `model` of the named case on its data.
pub fn model(case: &str, v: Value) -> Result<Model, String> {
    Ok(match case {
        "agriprice" => agriprice::model(&de(v)?),
        "avocado" => avocado::model(&de(v)?),
        "colour" => colour::model(&de(v)?),
        "cover" => cover::model(&de(v)?),
        "cutstock" => cutstock::model(&de(v)?),
        "economic" => economic::model(&de(v)?),
        "facility" => facility::model(&de(v)?),
        "factory" => factory::model(&de(v)?).0,
        "fit" => fit::model(&de(v)?, false),
        "food" => food::model(&de(v)?).0,
        "manpower" => manpower::model(&de(v)?).0,
        "network" => network::model(&de(v)?).0,
        "opencast" => opencast::model(&de(v)?),
        "playlist" => playlist::model(&de(v)?, false).0,
        "pooling" => pooling::model(&de(v)?, true).0,
        "portfolio" => portfolio::model(&de(v)?, None),
        "subset" => subset::model(&de(v)?, 1).0,
        "tsp" => tsp::model(&de(v)?).0,
        "workforce" => workforce::model(&de(v)?).0,
        _ => return Err(format!("unknown case {case}")),
    })
}

/// Name → position, for data that refer to entities by name.
pub(crate) struct Names(HashMap<String, usize>);

impl Names {
    pub(crate) fn new<'a>(names: impl IntoIterator<Item = &'a str>) -> Names {
        Names(names.into_iter().enumerate().map(|(k, s)| (s.to_string(), k)).collect())
    }

    pub(crate) fn at(&self, s: &str) -> usize {
        *self.0.get(s).unwrap_or_else(|| panic!("data name {s:?} is not defined"))
    }
}

/// Stock balance s_{t−1} + a_t = b_t + s_t for t = 0..T with s_{−1} = s0, and s_{T−1} = target:
/// the inventory rows that Williams's multi-period planning models share (columns given per t).
pub(crate) fn stock(m: &mut Model, s0: f64, a: &[usize], b: &[usize], s: &[usize], target: f64) {
    for t in 0..s.len() {
        let mut r = vec![(a[t], 1.0), (b[t], -1.0), (s[t], -1.0)];
        if t > 0 {
            r.push((s[t - 1], 1.0));
        }
        m.eq(r, if t == 0 { -s0 } else { 0.0 });
    }
    m.eq([(s[s.len() - 1], 1.0)], target);
}

/// Columns x_{r,k} for r in 0..nr, k in 0..nk, each made by `f`.
pub(crate) fn grid(
    m: &mut Model,
    nr: usize,
    nk: usize,
    mut f: impl FnMut(&mut Model, usize, usize) -> usize,
) -> Vec<Vec<usize>> {
    (0..nr).map(|r| (0..nk).map(|k| f(m, r, k)).collect()).collect()
}
