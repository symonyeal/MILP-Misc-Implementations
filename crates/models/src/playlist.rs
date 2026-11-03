//! Artist recommendation: hypergraph ranking, then selection by MILP. Source: Gurobi
//! modeling-examples `music_recommendation`, whose preference scores come from a LightFM matrix
//! factorisation; here they come from the hypergraph ranking of Zhou, Huang and Schölkopf (NIPS
//! 2006) on the hypergraph the user–artist plays induce (DHG `Hypergraph.from_bigraph` with artists as vertices).
//!
//! Scores. Artists are vertices; each user's set of played artists is a hyperedge of weight 1.
//! The chosen user's track counts, scaled to max 1, are y; f = (1 − α)(I − αΘ)⁻¹ y (`Hg::rank`),
//! min–max scaled to [0, 1] as the notebook scales its predictions.
//!
//! Selection over the `top` best-scored artists A (notebook: 1000), artist a with listeners ℓ_a,
//! countries c ⊆ A (a partition of A):
//!
//!   max Σ_a f_a x_a   s.t.  Σ_a ℓ_a x_a ≤ P N,   Σ_a x_a = N,   [w_c ≤ Σ_{a∈c} x_a,  Σ_c w_c ≥ C],
//!   x, w binary;
//!
//! the bracketed country rows are the notebook's second model. Columns: x by rank in A, then w_c.

use std::collections::BTreeMap;
use std::fmt::Write;

use hyper::Hg;
use milp::Model;
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Data {
    pub user: usize,
    pub alpha: f64,
    pub top: usize,
    pub n: f64,
    pub pmax: f64,
    pub cmin: f64,
    pub artists: Vec<Artist>,
    pub plays: Vec<(usize, usize, f64)>,
}

#[derive(Deserialize)]
pub struct Artist {
    pub name: String,
    pub country: String,
    pub listeners: f64,
}

/// f over all artists, scaled to [0, 1].
pub fn scores(d: &Data) -> Vec<f64> {
    let nv = d.artists.len();
    let mut by: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut y = vec![0.0; nv];
    for &(u, a, k) in &d.plays {
        by.entry(u).or_default().push(a);
        if u == d.user {
            y[a] = k;
        }
    }
    let ym = y.iter().fold(0.0, |a: f64, &b| a.max(b));
    assert!(ym > 0.0, "user {} has no plays", d.user);
    y.iter_mut().for_each(|v| *v /= ym);
    let f = Hg::new(nv, by.into_values()).rank(&y, d.alpha);
    let (lo, hi) = f.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    f.iter().map(|v| (v - lo) / (hi - lo)).collect()
}

/// The selection model over A (artist indices by score, best first), with country rows when `cty`.
pub fn model(d: &Data, cty: bool) -> (Model, Vec<usize>) {
    model_on(d, &scores(d), cty)
}

fn model_on(d: &Data, f: &[f64], cty: bool) -> (Model, Vec<usize>) {
    let mut a: Vec<usize> = (0..f.len()).collect();
    a.sort_by(|&p, &q| f[q].total_cmp(&f[p]).then(p.cmp(&q)));
    a.truncate(d.top);
    let mut m = Model::new(true);
    for &v in &a {
        m.bin(f[v]);
    }
    m.le(a.iter().enumerate().map(|(k, &v)| (k, d.artists[v].listeners)), d.pmax * d.n);
    m.eq((0..a.len()).map(|k| (k, 1.0)), d.n);
    if cty {
        let mut by: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (k, &v) in a.iter().enumerate() {
            by.entry(d.artists[v].country.as_str()).or_default().push(k);
        }
        let w: Vec<usize> = by.values().map(|_| m.bin(0.0)).collect();
        for (&wc, ks) in w.iter().zip(by.values()) {
            m.le([(wc, 1.0)].into_iter().chain(ks.iter().map(|&k| (k, -1.0))), 0.0);
        }
        m.ge(w.iter().map(|&j| (j, 1.0)), d.cmin);
    }
    (m, a)
}

pub fn run(d: &Data) -> Out {
    let f = scores(d);
    let mut text = String::new();
    let mut obj = Vec::new();
    let mut last = None;
    for cty in [false, true] {
        let (m, a) = model_on(d, &f, cty);
        let s = m.solve();
        obj.push(s.obj);
        if !s.x.is_empty() {
            let pick: Vec<usize> = (0..a.len()).filter(|&k| s.x[k] > 0.5).map(|k| a[k]).collect();
            let mut cs: Vec<&str> = pick.iter().map(|&v| d.artists[v].country.as_str()).collect();
            cs.sort();
            cs.dedup();
            let rule = if cty { format!("at least {} countries", d.cmin) } else { "no country rule".to_string() };
            let _ = writeln!(text, "{rule}: score {:.4}, {} artists from {} countries", s.obj, pick.len(), cs.len());
            let names: Vec<&str> = pick.iter().map(|&v| d.artists[v].name.as_str()).collect();
            let _ = writeln!(text, "  {}", names.join("; "));
        }
        last = Some((m, s));
    }
    let (m, s) = last.unwrap();
    Out::of(&m, &s, obj, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_matches_enumeration() {
        // Six artists, three countries, choose N = 3 with ≥ 2 countries under a listener budget.
        let ar = |c: &str, l: f64| Artist { name: String::new(), country: c.into(), listeners: l };
        let d = Data {
            user: 0,
            alpha: 0.5,
            top: 6,
            n: 3.0,
            pmax: 40.0,
            cmin: 2.0,
            artists: vec![ar("A", 50.0), ar("A", 10.0), ar("A", 30.0), ar("B", 60.0), ar("B", 20.0), ar("C", 40.0)],
            plays: vec![],
        };
        let f = [0.9, 0.8, 0.7, 0.95, 0.3, 0.6];
        let (m, a) = model_on(&d, &f, true);
        let s = m.solve();
        let mut best = f64::NEG_INFINITY;
        for mask in 0u32..64 {
            let pick: Vec<usize> = (0..6).filter(|&v| mask >> v & 1 == 1).collect();
            let mut cs: Vec<&str> = pick.iter().map(|&v| d.artists[v].country.as_str()).collect();
            cs.dedup();
            let l: f64 = pick.iter().map(|&v| d.artists[v].listeners).sum();
            if pick.len() == 3 && cs.len() >= 2 && l <= 120.0 {
                best = best.max(pick.iter().map(|&v| f[v]).sum());
            }
        }
        assert!((s.obj - best).abs() < 1e-9 && a.len() == 6);
    }
}
