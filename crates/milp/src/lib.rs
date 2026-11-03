//! A sparse mixed-integer linear program and its solve by HiGHS 1.12.0.
//!
//!   optimise  c0 + c'x   subject to  lo <= A x <= hi,  l <= x <= u,  x_j integer for j in J.
//!
//! Legend
//!   x ....... columns, x_j indexed 0..n          A ....... rows by CSR: p (row starts), j, a
//!   c, c0 ... objective and constant             lo, hi .. row bounds (±INF when open)
//!   l, u .... column bounds                      int ..... true on the integer columns J
//!   y ....... row duals of an LP, HiGHS sign: c - A'y is the reduced cost

mod highs;

pub use highs::version;

pub const INF: f64 = f64::INFINITY;

/// How a solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    Infeasible,
    Unbounded,
    /// A time, iteration or solution limit stopped the search; `x` holds the incumbent, if any.
    Limit,
    /// Any other HiGHS model status (load, presolve or solve error, interrupt).
    Error(i32),
}

/// One solve: status, objective c0 + c'x, the best bound, x, and y (LP only, else empty).
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

    /// The columns of row i.
    pub fn cols(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        self.j[self.p[i] as usize..self.p[i + 1] as usize].iter().map(|&j| j as usize)
    }

    /// A x.
    pub fn ax(&self, x: &[f64]) -> Vec<f64> {
        (0..self.m())
            .map(|i| (self.p[i] as usize..self.p[i + 1] as usize).map(|k| self.a[k] * x[self.j[k] as usize]).sum())
            .collect()
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

    /// Solve with HiGHS defaults.
    pub fn solve(&self) -> Sol {
        self.solve_with(&[])
    }

    /// Solve with HiGHS double options set by name, e.g. ("time_limit", 60.0).
    pub fn solve_with(&self, opts: &[(&str, f64)]) -> Sol {
        let (n, m) = (self.n(), self.m());
        let h = highs::H::new();
        for &(k, v) in opts {
            h.real(k, v);
        }
        let integ: Vec<i32> = self.int.iter().map(|&b| b as i32).collect();
        let mip = self.int.iter().any(|&b| b);
        let sense = if self.max { -1 } else { 1 };
        let mut s = Sol { st: Status::Error(2), obj: f64::NAN, bound: f64::NAN, x: vec![], y: vec![] };
        if !h.pass(sense, self.c0, &self.c, &self.l, &self.u, &self.lo, &self.hi, &self.p, &self.j, &self.a, &integ) {
            return s;
        }
        let ms = h.run();
        s.st = match ms {
            7 => Status::Optimal,
            8 => Status::Infeasible,
            9 | 10 => Status::Unbounded,
            13 | 14 | 16 => Status::Limit,
            _ => Status::Error(ms),
        };
        if matches!(s.st, Status::Optimal | Status::Limit) {
            let (x, y) = h.solution(n, m);
            s.obj = h.info("objective_function_value");
            s.bound = if mip { h.info("mip_dual_bound") } else { s.obj };
            s.x = x;
            if !mip {
                s.y = y;
            }
        }
        s
    }

    /// Hierarchical objectives in the sense of Gurobi's `setObjectiveN`: objectives (c_k, reltol_k)
    /// in priority order, each optimised over the solutions that degrade no earlier objective by more
    /// than max(reltol_k |z_k|, 1e-6) (Gurobi's MIP rule, ObjNAbsTol at its default 1e-6).
    /// OUTPUT the solution of each stage; the sequence stops at the first non-optimal stage.
    pub fn lex(&self, objs: &[(&[f64], f64)]) -> Vec<Sol> {
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
