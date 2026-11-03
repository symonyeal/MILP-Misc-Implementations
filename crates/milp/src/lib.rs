//! A sparse mixed-integer quadratically constrained program and its solve.
//!
//!   optimise  c0 + c'x + x'Q₀x   subject to  lo_i <= a_i'x + x'Q_i x <= hi_i,  l <= x <= u,
//!             x_j integer for j in J.
//!
//! Linear models go to HiGHS 1.12.0; models with any quadratic term go to SCIP 10, which solves
//! nonconvex ones to global optimality by spatial branch and bound. Either solver runs a linear
//! model (`solve_on`), so each can check the other.
//!
//! Legend
//!   x ....... columns, x_j indexed 0..n          A ....... rows by CSR: p (row starts), j, a
//!   c, c0 ... objective and constant             lo, hi .. row bounds (±INF when open)
//!   l, u .... column bounds                      int ..... true on the integer columns J
//!   qc ...... Q₀ as terms (j, k, v): v x_j x_k   qa ...... Q_i as terms (i, j, k, v)
//!   y ....... row duals of an LP solved by HiGHS, HiGHS sign: c - A'y is the reduced cost

mod highs;
mod scip;

pub use highs::version;

pub const INF: f64 = f64::INFINITY;

/// The solver a model is sent to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Highs,
    Scip,
}

/// How a solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    Infeasible,
    Unbounded,
    /// A time, iteration or solution limit stopped the search; `x` holds the incumbent, if any.
    Limit,
    /// Any other outcome (load, presolve or solve error, interrupt): the HiGHS model status, or
    /// -1 from SCIP.
    Error(i32),
}

/// One solve: status, objective, the best bound, x, and y (HiGHS LP only, else empty).
#[derive(Clone, Debug)]
pub struct Sol {
    pub st: Status,
    pub obj: f64,
    pub bound: f64,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub max: bool,
    pub c0: f64,
    pub c: Vec<f64>,
    pub l: Vec<f64>,
    pub u: Vec<f64>,
    pub int: Vec<bool>,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    pub p: Vec<i32>,
    pub j: Vec<i32>,
    pub a: Vec<f64>,
    pub qc: Vec<(usize, usize, f64)>,
    pub qa: Vec<(usize, usize, usize, f64)>,
    /// The solvers' row (and HiGHS's integrality) tolerance when set; both default to 1e-6. A
    /// nonlinear row's slack, or a big-M row's integrality slack, can carry into the objective; a
    /// tighter tolerance removes that, at a cost in time.
    pub feastol: Option<f64>,
}

impl Model {
    pub fn new(max: bool) -> Model {
        Model { max, p: vec![0], ..Model::default() }
    }

    pub fn n(&self) -> usize {
        self.c.len()
    }

    pub fn m(&self) -> usize {
        self.lo.len()
    }

    /// A column l <= x_j <= u with cost c; OUTPUT j.
    pub fn var(&mut self, l: f64, u: f64, c: f64) -> usize {
        self.c.push(c);
        self.l.push(l);
        self.u.push(u);
        self.int.push(false);
        self.n() - 1
    }

    /// An integer column l <= x_j <= u with cost c.
    pub fn int(&mut self, l: f64, u: f64, c: f64) -> usize {
        let j = self.var(l, u, c);
        self.int[j] = true;
        j
    }

    /// A binary column with cost c.
    pub fn bin(&mut self, c: f64) -> usize {
        self.int(0.0, 1.0, c)
    }

    /// A row lo <= Σ a_j x_j <= hi; repeated columns are summed and zero terms dropped.
    /// OUTPUT the row index i.
    pub fn row(&mut self, lo: f64, t: impl IntoIterator<Item = (usize, f64)>, hi: f64) -> usize {
        let mut t: Vec<(usize, f64)> = t.into_iter().collect();
        t.sort_by_key(|&(j, _)| j);
        let k0 = self.j.len();
        for (j, a) in t {
            assert!(j < self.n(), "row term on column {j} of {}", self.n());
            if self.j.len() > k0 && *self.j.last().unwrap() == j as i32 {
                *self.a.last_mut().unwrap() += a;
            } else {
                self.j.push(j as i32);
                self.a.push(a);
            }
        }
        let mut k = k0;
        for r in k0..self.j.len() {
            if self.a[r] != 0.0 {
                (self.j[k], self.a[k]) = (self.j[r], self.a[r]);
                k += 1;
            }
        }
        self.j.truncate(k);
        self.a.truncate(k);
        self.p.push(k as i32);
        self.lo.push(lo);
        self.hi.push(hi);
        self.m() - 1
    }

    pub fn le(&mut self, t: impl IntoIterator<Item = (usize, f64)>, b: f64) -> usize {
        self.row(-INF, t, b)
    }

    pub fn ge(&mut self, t: impl IntoIterator<Item = (usize, f64)>, b: f64) -> usize {
        self.row(b, t, INF)
    }

    pub fn eq(&mut self, t: impl IntoIterator<Item = (usize, f64)>, b: f64) -> usize {
        self.row(b, t, b)
    }

    /// A row lo <= a'x + Σ v x_j x_k <= hi, quadratic terms q = (j, k, v). OUTPUT i.
    pub fn qrow(
        &mut self,
        lo: f64,
        t: impl IntoIterator<Item = (usize, f64)>,
        q: impl IntoIterator<Item = (usize, usize, f64)>,
        hi: f64,
    ) -> usize {
        let i = self.row(lo, t, hi);
        for (j, k, v) in q {
            assert!(j.max(k) < self.n(), "quadratic term on column {} of {}", j.max(k), self.n());
            self.qa.push((i, j, k, v));
        }
        i
    }

    /// Adds v x_j x_k to the objective.
    pub fn qobj(&mut self, j: usize, k: usize, v: f64) {
        assert!(j.max(k) < self.n(), "quadratic term on column {} of {}", j.max(k), self.n());
        self.qc.push((j, k, v));
    }

    /// True when the model has no quadratic term.
    pub fn linear(&self) -> bool {
        self.qc.is_empty() && self.qa.is_empty()
    }

    /// The linear columns of row i.
    pub fn cols(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        self.j[self.p[i] as usize..self.p[i + 1] as usize].iter().map(|&j| j as usize)
    }

    /// Row values a_i'x + x'Q_i x.
    pub fn ax(&self, x: &[f64]) -> Vec<f64> {
        let mut r: Vec<f64> = (0..self.m())
            .map(|i| (self.p[i] as usize..self.p[i + 1] as usize).map(|k| self.a[k] * x[self.j[k] as usize]).sum())
            .collect();
        for &(i, j, k, v) in &self.qa {
            r[i] += v * x[j] * x[k];
        }
        r
    }

    /// The objective c0 + c'x + x'Q₀x.
    pub fn obj(&self, x: &[f64]) -> f64 {
        let l: f64 = self.c.iter().zip(x).map(|(c, x)| c * x).sum();
        self.c0 + l + self.qc.iter().map(|&(j, k, v)| v * x[j] * x[k]).sum::<f64>()
    }

    /// The largest violation by x of a bound, a row or integrality: the replay every solution passes.
    pub fn viol(&self, x: &[f64]) -> f64 {
        let col = (0..self.n()).map(|j| {
            let r = if self.int[j] { (x[j] - x[j].round()).abs() } else { 0.0 };
            r.max(self.l[j] - x[j]).max(x[j] - self.u[j])
        });
        let ax = self.ax(x);
        let row = (0..self.m()).map(|i| (self.lo[i] - ax[i]).max(ax[i] - self.hi[i]));
        col.chain(row).fold(0.0, |a, b| if b > a { b } else { a })
    }

    /// Solve on HiGHS when the model is linear, else on SCIP.
    pub fn solve(&self) -> Sol {
        self.solve_on(if self.linear() { Backend::Highs } else { Backend::Scip })
    }

    /// Solve on the named backend. HiGHS takes linear models and continuous QPs with a convex
    /// objective (positive semidefinite Q₀ when minimising), with no quadratic rows; a caller that
    /// knows its QP is convex sends it there for HiGHS's exact QP solver, since SCIP meets the
    /// objective's epigraph row only to its feasibility tolerance (1e-6 absolute).
    pub fn solve_on(&self, b: Backend) -> Sol {
        match b {
            Backend::Highs => {
                assert!(self.qa.is_empty(), "HiGHS takes no quadratic rows");
                assert!(self.qc.is_empty() || !self.int.iter().any(|&b| b), "HiGHS takes no integer QP");
                highs::solve(self)
            }
            Backend::Scip => scip::solve(self),
        }
    }

    /// Hierarchical objectives in the sense of Gurobi's `setObjectiveN`: linear objectives
    /// (c_k, reltol_k) in priority order, each optimised over the solutions that degrade no earlier
    /// objective by more than max(reltol_k |z_k|, 1e-6) (Gurobi's MIP rule, ObjNAbsTol at its
    /// default 1e-6). OUTPUT the solution of each stage; the sequence stops at the first
    /// non-optimal stage.
    pub fn lex(&self, objs: &[(&[f64], f64)]) -> Vec<Sol> {
        assert!(self.qc.is_empty(), "lex takes linear objectives");
        let mut m = self.clone();
        let mut out = Vec::new();
        for &(c, tol) in objs {
            m.c = c.to_vec();
            let s = m.solve();
            let ok = s.st == Status::Optimal;
            let z = s.obj - m.c0;
            out.push(s);
            if !ok {
                break;
            }
            let d = (tol * z.abs()).max(1e-6);
            let t = c.iter().enumerate().map(|(j, &cj)| (j, cj));
            if self.max {
                m.ge(t, z - d);
            } else {
                m.le(t, z + d);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_pinned() {
        assert_eq!(version(), "1.12.0");
    }

    #[test]
    fn lp_duals() {
        // min x + 2y, x + y >= 1, x - y <= 0: optimum x = y = 1/2, value 3/2, duals (3/2, -1/2).
        let mut m = Model::new(false);
        let (x, y) = (m.var(0.0, INF, 1.0), m.var(0.0, INF, 2.0));
        m.ge([(x, 1.0), (y, 1.0)], 1.0);
        m.le([(x, 1.0), (y, -1.0)], 0.0);
        let s = m.solve();
        assert_eq!(s.st, Status::Optimal);
        assert!((s.obj - 1.5).abs() < 1e-9);
        assert!((s.y[0] - 1.5).abs() < 1e-9 && (s.y[1] + 0.5).abs() < 1e-9);
        assert!(m.viol(&s.x) < 1e-9);
    }

    #[test]
    fn mip_and_outcomes() {
        // max 5a + 4b, 6a + 4b <= 24, a + 2b <= 6, integer: optimum a = 4, b = 0, value 20.
        let mut m = Model::new(true);
        let (a, b) = (m.int(0.0, INF, 5.0), m.int(0.0, INF, 4.0));
        m.le([(a, 6.0), (b, 4.0)], 24.0);
        m.le([(a, 1.0), (b, 2.0)], 6.0);
        let s = m.solve();
        assert_eq!(s.st, Status::Optimal);
        assert_eq!(s.obj, 20.0);
        assert!(s.y.is_empty());
        let mut f = m.clone();
        f.ge([(a, 1.0)], 5.0);
        assert_eq!(f.solve().st, Status::Infeasible);
        let mut g = Model::new(true);
        g.var(0.0, INF, 1.0);
        assert_eq!(g.solve().st, Status::Unbounded);
    }

    #[test]
    fn rows_merge_repeated_columns() {
        let mut m = Model::new(false);
        let (x, y) = (m.var(0.0, 1.0, 0.0), m.var(0.0, 1.0, 0.0));
        m.le([(y, 1.0), (x, 2.0), (y, -1.0), (x, 1.0)], 1.0);
        assert_eq!((m.j.clone(), m.a.clone(), m.p.clone()), (vec![0], vec![3.0], vec![0, 1]));
    }

    #[test]
    fn scip_global_optima() {
        // max xy, x + y <= 4, 0 <= x, y <= 4: a nonconvex objective, optimum 4 at (2, 2).
        let mut m = Model::new(true);
        let (x, y) = (m.var(0.0, 4.0, 0.0), m.var(0.0, 4.0, 0.0));
        m.le([(x, 1.0), (y, 1.0)], 4.0);
        m.qobj(x, y, 1.0);
        let s = m.solve();
        assert_eq!(s.st, Status::Optimal);
        assert!((s.obj - 4.0).abs() < 1e-6 && (s.x[x] - 2.0).abs() < 1e-4);
        // min x, x² >= 4, -1 <= x <= 3: a nonconvex feasible set [2, 3], optimum 2.
        let mut m = Model::new(false);
        let x = m.var(-1.0, 3.0, 1.0);
        m.qrow(4.0, [], [(x, x, 1.0)], INF);
        let s = m.solve();
        assert!((s.obj - 2.0).abs() < 1e-6 && m.viol(&s.x) < 1e-6);
        // min (x − 2.6)², x integer: x = 3, value 0.16.
        let mut m = Model::new(false);
        let x = m.int(-10.0, 10.0, -5.2);
        m.c0 = 6.76;
        m.qobj(x, x, 1.0);
        let s = m.solve();
        assert_eq!(s.x[x], 3.0);
        assert!((s.obj - 0.16).abs() < 1e-9 && (m.obj(&s.x) - s.obj).abs() < 1e-12);
    }

    #[test]
    fn convex_qp_on_both() {
        // min x² + xy + y² − 3x, −10 <= x, y <= 10: 2x + y = 3, x + 2y = 0, so (2, −1), value −3.
        let mut m = Model::new(false);
        let (x, y) = (m.var(-10.0, 10.0, -3.0), m.var(-10.0, 10.0, 0.0));
        m.qobj(x, x, 1.0);
        m.qobj(x, y, 1.0);
        m.qobj(y, y, 1.0);
        // Measured: HiGHS places x within 2e-7 (its QP optimality tolerance), SCIP's outer
        // approximation of the epigraph within 5e-4 (about √feastol); both values within 1e-6.
        for (b, tx) in [(Backend::Highs, 1e-6), (Backend::Scip, 1e-3)] {
            let s = m.solve_on(b);
            assert_eq!(s.st, Status::Optimal, "{b:?}");
            assert!((s.obj + 3.0).abs() < 1e-6 && (s.x[x] - 2.0).abs() < tx, "{b:?} {} {}", s.obj, s.x[x]);
        }
    }

    #[test]
    fn backends_agree_on_a_mip() {
        let mut m = Model::new(true);
        let (a, b) = (m.int(0.0, INF, 5.0), m.int(0.0, INF, 4.0));
        m.le([(a, 6.0), (b, 4.0)], 24.0);
        m.le([(a, 1.0), (b, 2.0)], 6.0);
        let (h, s) = (m.solve_on(Backend::Highs), m.solve_on(Backend::Scip));
        assert_eq!((h.st, s.st), (Status::Optimal, Status::Optimal));
        assert!((h.obj - s.obj).abs() < 1e-9 && s.obj == 20.0);
    }

    #[test]
    fn lex_degrades_by_reltol() {
        // min (x + y) then max y as min -y, x + y >= 10, 0 <= x, y <= 8, reltol 0.1:
        // stage 1 gives 10; stage 2 may use x + y <= 11, so y = 8.
        let mut m = Model::new(false);
        let (x, y) = (m.var(0.0, 8.0, 0.0), m.var(0.0, 8.0, 0.0));
        m.ge([(x, 1.0), (y, 1.0)], 10.0);
        let s = m.lex(&[(&[1.0, 1.0], 0.1), (&[0.0, -1.0], 0.0)]);
        assert!((s[0].obj - 10.0).abs() < 1e-9 && (s[1].obj + 8.0).abs() < 1e-9);
        assert!(s[1].x[x] + s[1].x[y] <= 11.0 + 1e-6);
    }
}
