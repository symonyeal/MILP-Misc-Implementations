//! Refinery optimisation: H. P. Williams, *Model Building in Mathematical Programming* (5th ed.,
//! 2013), problem "Refinery optimization". Source: Gurobi modeling-examples `refinery`; stream
//! names are the notebook's.
//!
//! Crudes distil into naphthas, oils and residuum; naphthas may be reformed into reformed
//! gasoline, oils cracked into cracked gasoline and cracked oil, residuum turned into lube oil.
//! The motor fuels blend naphthas and gasolines under octane floors, jet fuel blends oils and
//! residuum under a vapour-pressure ceiling, and fuel oil takes fixed proportions. Every row is a
//! yield, a capacity, a stream balance or a quality bound (the notebook's constraint names in
//! brackets below); max Σ profit × end product. Columns are named; `col` maps a name to its index.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub buy_limit: Vec<f64>,
    pub lbo_min: f64,
    pub lbo_max: f64,
    pub distill_cap: f64,
    pub reform_cap: f64,
    pub crack_cap: f64,
    pub end_product_profit: BTreeMap<String, f64>,
    pub blending_coefficients: BTreeMap<String, f64>,
    pub lube_oil_factor: f64,
    pub pmf_rmf_ratio: f64,
    pub octance_number_coefficients: BTreeMap<String, f64>,
    pub octance_number_fuel: BTreeMap<String, f64>,
    pub vapor_pressure_constants: Vec<f64>,
    pub reforming_splitting_coefficients: BTreeMap<String, f64>,
    pub distillation: BTreeMap<String, Vec<f64>>,
    pub cracking: Vec<(String, String, f64)>,
}

const END: [&str; 5] = ["Premium_fuel", "Regular_fuel", "Jet_fuel", "Fuel_oil", "Lube_oil"];
const DIST: [&str; 6] = ["Light_naphtha", "Medium_naphtha", "Heavy_naphtha", "Light_oil", "Heavy_oil", "Residuum"];
const NAPH: [&str; 3] = ["Light_naphtha", "Medium_naphtha", "Heavy_naphtha"];
const OILS: [&str; 2] = ["Light_oil", "Heavy_oil"];
const CRACK: [&str; 2] = ["Cracked_gasoline", "Cracked_oil"];
const MOTOR: [&str; 5] = ["Light_naphtha", "Medium_naphtha", "Heavy_naphtha", "Reformed_gasoline", "Cracked_gasoline"];
const JET: [&str; 4] = ["Light_oil", "Heavy_oil", "Residuum", "Cracked_oil"];

/// The model and its column names.
pub fn model(d: &Data) -> (Model, HashMap<String, usize>) {
    let mut m = Model::new(true);
    let mut col: HashMap<String, usize> = HashMap::new();
    let mut add = |m: &mut Model, name: String, l: f64, u: f64, c: f64| {
        col.insert(name, m.var(l, u, c));
    };
    for (k, &u) in d.buy_limit.iter().enumerate() {
        add(&mut m, format!("crude{}", k + 1), 0.0, u, 0.0);
    }
    for e in END {
        let (l, u) = if e == "Lube_oil" { (d.lbo_min, d.lbo_max) } else { (0.0, INF) };
        add(&mut m, format!("end {e}"), l, u, d.end_product_profit[e]);
    }
    let names = DIST.map(|s| format!("dist {s}")).into_iter().chain(NAPH.map(|s| format!("reform {s}")));
    let names = names
        .chain(["reformed".to_string()])
        .chain(OILS.map(|s| format!("crack {s}")))
        .chain(CRACK.map(|s| format!("cracked {s}")));
    let names = names
        .chain(MOTOR.map(|s| format!("regular {s}")))
        .chain(MOTOR.map(|s| format!("premium {s}")))
        .chain(JET.map(|s| format!("jet {s}")));
    for nm in names.chain(["lube residuum".to_string()]) {
        add(&mut m, nm, 0.0, INF, 0.0);
    }
    let c = |s: &str| *col.get(s).unwrap_or_else(|| panic!("no column {s}"));
    let crudes: Vec<usize> = (0..d.buy_limit.len()).map(|k| c(&format!("crude{}", k + 1))).collect();
    // Capacities [Distill_cap, Reform_cap, Crack_cap].
    m.le(crudes.iter().map(|&j| (j, 1.0)), d.distill_cap);
    m.le(NAPH.iter().map(|n| (c(&format!("reform {n}")), 1.0)), d.reform_cap);
    m.le(OILS.iter().map(|o| (c(&format!("crack {o}")), 1.0)), d.crack_cap);
    // Yields [Splitting_distillation, _reforming, _cracking, _lube_oil].
    for p in DIST {
        let y = crudes.iter().enumerate().map(|(k, &j)| (j, d.distillation[p][k]));
        m.eq(y.chain([(c(&format!("dist {p}")), -1.0)]), 0.0);
    }
    let y = NAPH.iter().map(|n| (c(&format!("reform {n}")), d.reforming_splitting_coefficients[*n]));
    m.eq(y.chain([(c("reformed"), -1.0)]), 0.0);
    for out in CRACK {
        let y = d.cracking.iter().filter(|t| t.1 == out).map(|(o, _, v)| (c(&format!("crack {o}")), *v));
        m.eq(y.chain([(c(&format!("cracked {out}")), -1.0)]), 0.0);
    }
    m.eq([(c("lube residuum"), d.lube_oil_factor), (c("end Lube_oil"), -1.0)], 0.0);
    // Blends [Blending_premium_fuel, _regular_fuel, Continuity_jet_fuel].
    for (fuel, pre) in [("Premium_fuel", "premium"), ("Regular_fuel", "regular")] {
        m.eq(MOTOR.iter().map(|s| (c(&format!("{pre} {s}")), 1.0)).chain([(c(&format!("end {fuel}")), -1.0)]), 0.0);
    }
    m.eq(JET.iter().map(|s| (c(&format!("jet {s}")), 1.0)).chain([(c("end Jet_fuel"), -1.0)]), 0.0);
    // Stream balances [Continuity_naphtha, Fixed_proportion_*, Continuity_cracked/reformed_gasoline].
    for n in NAPH {
        let u = [format!("reform {n}"), format!("regular {n}"), format!("premium {n}")].map(|s| (c(&s), 1.0));
        m.eq(u.into_iter().chain([(c(&format!("dist {n}")), -1.0)]), 0.0);
    }
    let fo = c("end Fuel_oil");
    for (stream, uses, src) in [
        ("Light_oil", vec!["crack Light_oil", "jet Light_oil"], "dist Light_oil"),
        ("Heavy_oil", vec!["crack Heavy_oil", "jet Heavy_oil"], "dist Heavy_oil"),
        ("Cracked_oil", vec!["jet Cracked_oil"], "cracked Cracked_oil"),
        ("Residuum", vec!["lube residuum", "jet Residuum"], "dist Residuum"),
    ] {
        let u = uses.into_iter().map(|s| (c(s), 1.0)).chain([(fo, d.blending_coefficients[stream]), (c(src), -1.0)]);
        m.eq(u, 0.0);
    }
    for g in ["Cracked_gasoline", "Reformed_gasoline"] {
        let src = if g == "Cracked_gasoline" { c("cracked Cracked_gasoline") } else { c("reformed") };
        m.eq([(c(&format!("regular {g}")), 1.0), (c(&format!("premium {g}")), 1.0), (src, -1.0)], 0.0);
    }
    // Quality [Prem2reg_prop, Octane_tol_*, Vapor_pressure_tol].
    m.ge([(c("end Premium_fuel"), 1.0), (c("end Regular_fuel"), -d.pmf_rmf_ratio)], 0.0);
    for (fuel, pre) in [("Regular_fuel", "regular"), ("Premium_fuel", "premium")] {
        let o = MOTOR.iter().map(|s| (c(&format!("{pre} {s}")), d.octance_number_coefficients[*s]));
        m.ge(o.chain([(c(&format!("end {fuel}")), -d.octance_number_fuel[fuel])]), 0.0);
    }
    let v = &d.vapor_pressure_constants;
    let jets = [("jet Light_oil", 1.0), ("jet Heavy_oil", v[0]), ("jet Cracked_oil", v[1]), ("jet Residuum", v[2])];
    m.le(jets.into_iter().map(|(s, w)| (c(s), w)).chain([(c("end Jet_fuel"), -1.0)]), 0.0);
    (m, col)
}

pub fn run(d: &Data) -> Out {
    let (m, col) = model(d);
    let s = m.solve();
    let mut text = String::new();
    if !s.x.is_empty() {
        let _ = writeln!(text, "profit {:.4}", s.obj);
        for e in END {
            let _ = writeln!(text, "  {e:13} {:10.2}", s.x[col[&format!("end {e}")]]);
        }
    }
    Out::of(&m, &s, vec![s.obj], text)
}
