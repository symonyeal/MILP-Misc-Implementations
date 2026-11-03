//! Railway dispatching: trains on single-track lines with stations of limited capacity, scheduled
//! to minimize total delay. Source: Gurobi modeling-examples `railway_dispatching` (datasets
//! `linear`, `xshape`, `yshape`; the notebook runs `linear`).
//!
//! Train i enters resource r of its route at t_ir (its first at time 0) and the route's end at f_i;
//! d_r is r's running or dwell time and κ_r its capacity. With u the resource after r on i's route
//! and v the one after r on j's (or f when r is last), for each pair i, j sharing r exactly one of
//!
//!   y_ij: t_jr ≥ t_iu (j enters after i leaves),   y_ji: t_ir ≥ t_jv,
//!   x_ij: t_jv ≥ t_ir and t_iu ≥ t_jr (they overlap),
//!
//! holds, by big-M rows. At κ_r = 1 no pair may overlap; at κ_r > 1 every κ_r + 1 trains have a
//! pair that does not, Σ x ≤ C(κ_r + 1, 2) − 1 (pairwise overlapping intervals share a point).
//!   t_i,next(r) − t_ir ≥ d_r,   f_i − t_i,last = d_last;   min Σ_i (f_i − Σ_{r ∈ route i} d_r).
//!
//! The notebook's M, `sum(duration) * len(trains)`, sums the duration dict's keys (resource ids),
//! not its values; the port uses the durations. Neither M binds here: every optimal time is at most
//! the longest route plus the optimal total delay, which is below both (checked by the test).

use std::fmt::Write;

use milp::{INF, Model};
use serde::Deserialize;

use crate::Out;

#[derive(Deserialize)]
pub struct Resource {
    pub id: u32,
    pub duration: f64,
    pub capacity: usize,
}

#[derive(Deserialize)]
pub struct Train {
    pub id: String,
    /// Resource ids from the train's start.
    pub route: Vec<u32>,
}

#[derive(Deserialize)]
pub struct Data {
    pub resources: Vec<Resource>,
    pub trains: Vec<Train>,
}

/// The big-M: total resource time, once per train.
pub fn big_m(d: &Data) -> f64 {
    d.resources.iter().map(|r| r.duration).sum::<f64>() * d.trains.len() as f64
}

/// The model and, per train, the columns t_ir along its route followed by f_i.
pub fn model(d: &Data) -> (Model, Vec<Vec<usize>>) {
    let res = |id: u32| d.resources.iter().find(|r| r.id == id).expect("route resource");
    let mut m = Model::new(false);
    let mut t = Vec::new();
    for tr in &d.trains {
        let mut c: Vec<usize> = tr.route.iter().map(|_| m.var(0.0, INF, 0.0)).collect();
        m.u[c[0]] = 0.0;
        let f = m.var(0.0, INF, 1.0);
        m.c0 -= tr.route.iter().map(|&r| res(r).duration).sum::<f64>();
        c.push(f);
        let n = tr.route.len();
        for k in 0..n {
            let dk = res(tr.route[k]).duration;
            if k + 1 < n {
                m.ge([(c[k + 1], 1.0), (c[k], -1.0)], dk);
            } else {
                m.eq([(c[n], 1.0), (c[k], -1.0)], dk);
            }
        }
        t.push(c);
    }
    let big = big_m(d);
    // At 1e-6 a binary of 1 − 1e-6 times M slackens a row by 2e-4 (measured: 46.999999 on `linear`).
    m.feastol = Some(1e-9);
    for r in &d.resources {
        // (train, enter column, leave column) for each train using r.
        let on: Vec<(usize, usize, usize)> = d
            .trains
            .iter()
            .enumerate()
            .filter_map(|(i, tr)| tr.route.iter().position(|&q| q == r.id).map(|k| (i, t[i][k], t[i][k + 1])))
            .collect();
        let mut x = vec![vec![usize::MAX; on.len()]; on.len()];
        for a in 0..on.len() {
            for b in a + 1..on.len() {
                let ((_, ir, iu), (_, jr, jv)) = (on[a], on[b]);
                let yab = m.bin(0.0);
                m.ge([(jr, 1.0), (iu, -1.0), (yab, -big)], -big);
                let yba = m.bin(0.0);
                m.ge([(ir, 1.0), (jv, -1.0), (yba, -big)], -big);
                let xab = m.bin(0.0);
                m.ge([(jv, 1.0), (ir, -1.0), (xab, -big)], -big);
                m.ge([(iu, 1.0), (jr, -1.0), (xab, -big)], -big);
                m.eq([(yab, 1.0), (yba, 1.0), (xab, 1.0)], 1.0);
                x[a][b] = xab;
            }
        }
        let k = r.capacity;
        if on.len() <= k {
            continue;
        }
        if k == 1 {
            for (a, row) in x.iter().enumerate() {
                for &j in &row[a + 1..] {
                    m.u[j] = 0.0;
                }
            }
        } else {
            for s in subsets(on.len(), k + 1) {
                let pairs = s.iter().enumerate().flat_map(|(p, &a)| s[p + 1..].iter().map(move |&b| (a, b)));
                m.le(pairs.map(|(a, b)| (x[a][b], 1.0)), ((k + 1) * k / 2 - 1) as f64);
            }
        }
    }
    (m, t)
}

/// The k-subsets of 0..n in lexicographic order.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut s: Vec<usize> = (0..k).collect();
    loop {
        out.push(s.clone());
        let Some(i) = (0..k).rev().find(|&i| s[i] < n - k + i) else { return out };
        s[i] += 1;
        for j in i + 1..k {
            s[j] = s[j - 1] + 1;
        }
    }
}

pub fn run(d: &Data) -> Out {
    let (m, t) = model(d);
    let s = m.solve();
    let mut text = String::new();
    let _ = writeln!(text, "total delay {:.0}", s.obj);
    for (tr, c) in d.trains.iter().zip(&t) {
        let times: Vec<String> = c.iter().map(|&j| format!("{:.0}", s.x[j])).collect();
        let _ = writeln!(text, "  {} {}", tr.id, times.join(" "));
    }
    Out::of(&m, &s, vec![s.obj], text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsets_count() {
        assert_eq!(subsets(5, 3).len(), 10);
        assert_eq!(subsets(3, 3), vec![vec![0, 1, 2]]);
    }

    /// An optimal schedule ends each train by its route time plus the optimal total delay; when
    /// that is below M (and below the notebook's 6 · Σ ids = 126 on `linear`), M cuts no better
    /// schedule.
    #[test]
    fn big_m_does_not_bind() {
        for f in ["railway_linear", "railway_xshape", "railway_yshape"] {
            let p = format!("{}/../../data/{f}.json", env!("CARGO_MANIFEST_DIR"));
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
            let d: Data = serde_json::from_value(v["data"].clone()).unwrap();
            let z = model(&d).0.solve().obj;
            let dur = |id: u32| d.resources.iter().find(|r| r.id == id).unwrap().duration;
            let long = d.trains.iter().map(|t| t.route.iter().map(|&r| dur(r)).sum::<f64>()).fold(0.0, f64::max);
            assert!(long + z < big_m(&d), "{f}: {long} + {z} against {}", big_m(&d));
            if f == "railway_linear" {
                assert!(long + z < 126.0);
            }
        }
    }
}
