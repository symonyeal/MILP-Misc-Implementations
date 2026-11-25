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
pub mod aviation;
pub mod avocado;
pub mod battery;
pub mod campaign;
pub mod colour;
pub mod comparison;
pub mod constraint;
pub mod cover;
pub mod covid;
pub mod cutstock;
pub mod dea;
pub mod decentral;
pub mod drone;
pub mod economic;
pub mod facility;
pub mod factory;
pub mod farm;
pub mod fit;
pub mod folding;
pub mod food;
pub mod lineup;
pub mod logic;
pub mod luggage;
pub mod manpower;
pub mod market;
pub mod median;
pub mod milk;
pub mod mining;
pub mod network;
pub mod opencast;
pub mod playlist;
pub mod pooling;
pub mod portfolio;
pub mod power;
pub mod railway;
pub mod rap;
pub mod refinery;
pub mod relief;
pub mod rental;
pub mod subset;
pub mod techs;
pub mod tracking;
pub mod trucks;
pub mod tsp;
pub mod uc;
pub mod wind;
pub mod workforce;
pub mod yields;

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

// Each case's JSON schema and its two operations share one registration.
struct Case {
    name: &'static str,
    run: fn(Value) -> Result<Out, String>,
    model: fn(Value) -> Result<Model, String>,
}

const CASES: &[Case] = &[
    Case { name: "agriprice", run: |v| Ok(agriprice::run(&de(v)?)), model: |v| Ok(agriprice::model(&de(v)?)) },
    Case { name: "avocado", run: |v| Ok(avocado::run(&de(v)?)), model: |v| Ok(avocado::model(&de(v)?)) },
    Case { name: "aviation", run: |v| Ok(aviation::run(&de(v)?)), model: |v| Ok(aviation::model(&de(v)?, false).0) },
    Case { name: "battery", run: |v| Ok(battery::run(&de(v)?)), model: |v| Ok(battery::model(&de(v)?, None)) },
    Case { name: "trucks", run: |v| Ok(trucks::run(&de(v)?)), model: |v| Ok(trucks::model(&de(v)?)) },
    Case { name: "campaign", run: |v| Ok(campaign::run(&de(v)?)), model: |v| Ok(campaign::model(&de(v)?)) },
    Case { name: "colour", run: |v| Ok(colour::run(&de(v)?)), model: |v| Ok(colour::model(&de(v)?)) },
    Case { name: "comparison", run: |v| Ok(comparison::run(&de(v)?)), model: |v| Ok(comparison::model(&de(v)?)) },
    Case { name: "constraint", run: |v| Ok(constraint::run(&de(v)?)), model: |v| Ok(constraint::model(&de(v)?)) },
    Case { name: "cover", run: |v| Ok(cover::run(&de(v)?)), model: |v| Ok(cover::model(&de(v)?)) },
    Case { name: "covid", run: |v| Ok(covid::run(&de(v)?)), model: |v| Ok(covid::model(&de(v)?)) },
    Case { name: "cutstock", run: |v| Ok(cutstock::run(&de(v)?)), model: |v| Ok(cutstock::model(&de(v)?)) },
    Case { name: "dea", run: |v| Ok(dea::run(&de(v)?)), model: |v| Ok(dea::model(&de(v)?, 0)) },
    Case { name: "decentral", run: |v| Ok(decentral::run(&de(v)?)), model: |v| Ok(decentral::model(&de(v)?)) },
    Case { name: "drone", run: |v| Ok(drone::run(&de(v)?)), model: |v| Ok(drone::model(&de(v)?, false)) },
    Case { name: "economic", run: |v| Ok(economic::run(&de(v)?)), model: |v| Ok(economic::model(&de(v)?)) },
    Case { name: "facility", run: |v| Ok(facility::run(&de(v)?)), model: |v| Ok(facility::model(&de(v)?)) },
    Case { name: "farm", run: |v| Ok(farm::run(&de(v)?)), model: |v| Ok(farm::model(&de(v)?)) },
    Case { name: "factory", run: |v| Ok(factory::run(&de(v)?)), model: |v| Ok(factory::model(&de(v)?).0) },
    Case { name: "fit", run: |v| Ok(fit::run(&de(v)?)), model: |v| Ok(fit::model(&de(v)?, false)) },
    Case { name: "folding", run: |v| Ok(folding::run(&de(v)?)), model: |v| Ok(folding::model(&de(v)?).0) },
    Case { name: "food", run: |v| Ok(food::run(&de(v)?)), model: |v| Ok(food::model(&de(v)?).0) },
    Case { name: "lineup", run: |v| Ok(lineup::run(&de(v)?)), model: |v| Ok(lineup::model(&de(v)?).0) },
    Case { name: "logic", run: |v| Ok(logic::run(&de(v)?)), model: |v| Ok(logic::model(&de(v)?)) },
    Case { name: "luggage", run: |v| Ok(luggage::run(&de(v)?)), model: |v| Ok(luggage::model(&de(v)?).0) },
    Case { name: "manpower", run: |v| Ok(manpower::run(&de(v)?)), model: |v| Ok(manpower::model(&de(v)?).0) },
    Case { name: "market", run: |v| Ok(market::run(&de(v)?)), model: |v| Ok(market::model(&de(v)?)) },
    Case { name: "median", run: |v| Ok(median::run(&de(v)?)), model: |v| Ok(median::model(&de(v)?)) },
    Case { name: "milk", run: |v| Ok(milk::run(&de(v)?)), model: |v| Ok(milk::model(&de(v)?).0) },
    Case { name: "mining", run: |v| Ok(mining::run(&de(v)?)), model: |v| Ok(mining::model(&de(v)?).0) },
    Case { name: "network", run: |v| Ok(network::run(&de(v)?)), model: |v| Ok(network::model(&de(v)?).0) },
    Case { name: "opencast", run: |v| Ok(opencast::run(&de(v)?)), model: |v| Ok(opencast::model(&de(v)?)) },
    Case { name: "playlist", run: |v| Ok(playlist::run(&de(v)?)), model: |v| Ok(playlist::model(&de(v)?, false).0) },
    Case { name: "pooling", run: |v| Ok(pooling::run(&de(v)?)), model: |v| Ok(pooling::model(&de(v)?, true).0) },
    Case { name: "portfolio", run: |v| Ok(portfolio::run(&de(v)?)), model: |v| Ok(portfolio::model(&de(v)?, None)) },
    Case { name: "power", run: |v| Ok(power::run(&de(v)?)), model: |v| Ok(power::model(&de(v)?).0) },
    Case { name: "railway", run: |v| Ok(railway::run(&de(v)?)), model: |v| Ok(railway::model(&de(v)?).0) },
    Case { name: "rap", run: |v| Ok(rap::run(&de(v)?)), model: |v| Ok(rap::model(&de(v)?)) },
    Case { name: "refinery", run: |v| Ok(refinery::run(&de(v)?)), model: |v| Ok(refinery::model(&de(v)?).0) },
    Case { name: "relief", run: |v| Ok(relief::run(&de(v)?)), model: |v| Ok(relief::model(&de(v)?)) },
    Case { name: "rental", run: |v| Ok(rental::run(&de(v)?)), model: |v| Ok(rental::model(&de(v)?, false)) },
    Case { name: "subset", run: |v| Ok(subset::run(&de(v)?)), model: |v| Ok(subset::model(&de(v)?, 1).0) },
    Case { name: "techs", run: |v| Ok(techs::run(&de(v)?)), model: |v| Ok(techs::model(&de(v)?).0) },
    Case { name: "tracking", run: |v| Ok(tracking::run(&de(v)?)), model: |v| Ok(tracking::model(&de(v)?, true)) },
    Case { name: "tsp", run: |v| Ok(tsp::run(&de(v)?)), model: |v| Ok(tsp::model(&de(v)?).0) },
    Case { name: "uc", run: |v| Ok(uc::run(&de(v)?)), model: |v| Ok(uc::model(&de(v)?).0) },
    Case { name: "wind", run: |v| Ok(wind::run(&de(v)?)), model: |v| Ok(wind::model(&de(v)?)) },
    Case { name: "workforce", run: |v| Ok(workforce::run(&de(v)?)), model: |v| Ok(workforce::model(&de(v)?).0) },
    Case { name: "yields", run: |v| Ok(yields::run(&de(v)?)), model: |v| Ok(yields::model(&de(v)?, &[]).0) },
];

/// `run` of the named case on its data.
pub fn run(case: &str, v: Value) -> Result<Out, String> {
    let entry = CASES.iter().find(|entry| entry.name == case).ok_or_else(|| format!("unknown case {case}"))?;
    (entry.run)(v)
}

/// `model` of the named case on its data.
pub fn model(case: &str, v: Value) -> Result<Model, String> {
    let entry = CASES.iter().find(|entry| entry.name == case).ok_or_else(|| format!("unknown case {case}"))?;
    (entry.model)(v)
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
