//! The HiGHS C API calls this crate makes (`highs/interfaces/highs_c_api.h`, v1.12.0).

use std::ffi::{CStr, CString, c_char, c_void};

pub type Hi = i32;

unsafe extern "C" {
    fn Highs_create() -> *mut c_void;
    fn Highs_destroy(h: *mut c_void);
    fn Highs_version() -> *const c_char;
    fn Highs_passModel(
        h: *mut c_void,
        n: Hi,
        m: Hi,
        nnz: Hi,
        q_nnz: Hi,
        a_format: Hi,
        q_format: Hi,
        sense: Hi,
        offset: f64,
        c: *const f64,
        l: *const f64,
        u: *const f64,
        lo: *const f64,
        hi: *const f64,
        p: *const Hi,
        j: *const Hi,
        a: *const f64,
        qp: *const Hi,
        qj: *const Hi,
        qa: *const f64,
        integ: *const Hi,
    ) -> Hi;
    fn Highs_setBoolOptionValue(h: *mut c_void, k: *const c_char, v: Hi) -> Hi;
    fn Highs_run(h: *mut c_void) -> Hi;
    fn Highs_getModelStatus(h: *const c_void) -> Hi;
    fn Highs_getSolution(h: *const c_void, x: *mut f64, z: *mut f64, r: *mut f64, y: *mut f64) -> Hi;
    fn Highs_getDoubleInfoValue(h: *const c_void, k: *const c_char, v: *mut f64) -> Hi;
    fn Highs_setDoubleOptionValue(h: *mut c_void, k: *const c_char, v: f64) -> Hi;
}

/// `kHighsMatrixFormatRowwise`.
pub const ROWWISE: Hi = 2;

/// The linked HiGHS version, e.g. "1.12.0".
pub fn version() -> String {
    // SAFETY: Highs_version returns a static NUL-terminated string.
    unsafe { CStr::from_ptr(Highs_version()) }.to_string_lossy().into_owned()
}

/// An owned HiGHS instance, destroyed on drop.
pub struct H(*mut c_void);

impl Drop for H {
    fn drop(&mut self) {
        // SAFETY: created by Highs_create and destroyed once.
        unsafe { Highs_destroy(self.0) }
    }
}

impl H {
    pub fn new() -> H {
        // SAFETY: Highs_create returns an owned handle.
        let h = H(unsafe { Highs_create() });
        assert!(!h.0.is_null(), "HiGHS allocation failed");
        // MILP_LOG set: show the solver log (diagnosis only).
        h.flag("output_flag", std::env::var_os("MILP_LOG").is_some());
        // Relative MIP gap 1e-9, not HiGHS's 1e-4: "optimal" means optimal, not within 0.01%
        // (SCIP's default gap is already 0).
        h.real("mip_rel_gap", 1e-9);
        h
    }

    pub fn real(&self, k: &str, v: f64) {
        let k = CString::new(k).unwrap();
        // SAFETY: live handle, NUL-terminated key.
        unsafe { Highs_setDoubleOptionValue(self.0, k.as_ptr(), v) };
    }

    pub fn flag(&self, k: &str, v: bool) {
        let k = CString::new(k).unwrap();
        // SAFETY: live handle, NUL-terminated key.
        unsafe { Highs_setBoolOptionValue(self.0, k.as_ptr(), v as Hi) };
    }

    /// Highs_passModel; every slice must have the length its argument count states. The Hessian
    /// q = (start, index, value) is lower triangular by column (`kHighsHessianFormatTriangular`),
    /// the objective c'x + ½x'Qx; `integ` empty means all continuous.
    /// OUTPUT false if HiGHS rejects the model.
    pub fn pass(
        &self,
        sense: Hi,
        c0: f64,
        c: &[f64],
        l: &[f64],
        u: &[f64],
        lo: &[f64],
        hi: &[f64],
        p: &[Hi],
        j: &[Hi],
        a: &[f64],
        integ: &[Hi],
        q: Option<(&[Hi], &[Hi], &[f64])>,
    ) -> bool {
        let (qp, qi, qv) = q.unwrap_or((&[], &[], &[]));
        let ptr = |v: &[Hi]| if v.is_empty() { std::ptr::null() } else { v.as_ptr() };
        // SAFETY: every array outlives the call and has the length HiGHS reads.
        let st = unsafe {
            Highs_passModel(
                self.0,
                c.len() as Hi,
                lo.len() as Hi,
                a.len() as Hi,
                qv.len() as Hi,
                ROWWISE,
                if q.is_some() { 1 } else { 0 },
                sense,
                c0,
                c.as_ptr(),
                l.as_ptr(),
                u.as_ptr(),
                lo.as_ptr(),
                hi.as_ptr(),
                p.as_ptr(),
                j.as_ptr(),
                a.as_ptr(),
                ptr(qp),
                ptr(qi),
                if qv.is_empty() { std::ptr::null() } else { qv.as_ptr() },
                ptr(integ),
            )
        };
        st != -1
    }

    /// Highs_run, then the model status.
    pub fn run(&self) -> Hi {
        // SAFETY: live handle.
        unsafe {
            Highs_run(self.0);
            Highs_getModelStatus(self.0)
        }
    }

    /// (x, y): column values and row duals.
    pub fn solution(&self, n: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
        let (mut x, mut z, mut r, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; m], vec![0.0; m]);
        // SAFETY: live handle, arrays sized to the model.
        unsafe { Highs_getSolution(self.0, x.as_mut_ptr(), z.as_mut_ptr(), r.as_mut_ptr(), y.as_mut_ptr()) };
        (x, y)
    }

    pub fn info(&self, k: &str) -> f64 {
        let k = CString::new(k).unwrap();
        let mut v = 0.0;
        // SAFETY: live handle, valid out pointer.
        unsafe { Highs_getDoubleInfoValue(self.0, k.as_ptr(), &mut v) };
        v
    }
}

/// A linear model, or a continuous QP, solved by HiGHS with its defaults.
///
/// A QP's objective is first scaled by 2^k so that its largest Hessian entry lies in [1, 2):
/// HiGHS 1.12's QP solver cycled without progress on a covariance Hessian with entries 7e-8 to
/// 1e-3 (measured on the portfolio case) and converged once scaled. A power of two scales exactly,
/// so the objective and duals are unscaled without rounding.
pub fn solve(m: &crate::Model) -> crate::Sol {
    use crate::{Sol, Status};
    let h = H::new();
    if let Some(t) = m.feastol {
        h.real("primal_feasibility_tolerance", t);
        h.real("mip_feasibility_tolerance", t);
    }
    let mip = m.int.iter().any(|&b| b);
    let integ: Vec<i32> = if mip { m.int.iter().map(|&b| b as i32).collect() } else { vec![] };
    let mut q = hessian(m);
    let w = q.as_ref().map_or(1.0, |(_, _, v)| {
        let top = v.iter().fold(0.0, |a: f64, b| a.max(b.abs()));
        if top > 0.0 { 2f64.powi(-(top.log2().floor() as i32)) } else { 1.0 }
    });
    let c: Vec<f64> = m.c.iter().map(|c| c * w).collect();
    if let Some((_, _, v)) = q.as_mut() {
        v.iter_mut().for_each(|x| *x *= w);
    }
    let sense = if m.max { -1 } else { 1 };
    let mut s = Sol { st: Status::Error(2), obj: f64::NAN, bound: f64::NAN, x: vec![], y: vec![] };
    let qr = q.as_ref().map(|(a, b, c)| (&a[..], &b[..], &c[..]));
    if !h.pass(sense, m.c0 * w, &c, &m.l, &m.u, &m.lo, &m.hi, &m.p, &m.j, &m.a, &integ, qr) {
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
        let (x, y) = h.solution(m.n(), m.m());
        s.obj = h.info("objective_function_value") / w;
        s.bound = if mip { h.info("mip_dual_bound") / w } else { s.obj };
        s.x = x;
        if !mip {
            s.y = y.iter().map(|v| v / w).collect();
        }
    }
    s
}

/// Q₀ as HiGHS's lower-triangular Hessian: v x_j x_k adds 2v to Q_jj when j = k, else v to
/// Q_{max(j,k), min(j,k)}, so that ½x'Qx = Σ v x_j x_k. None for a linear objective.
fn hessian(m: &crate::Model) -> Option<(Vec<Hi>, Vec<Hi>, Vec<f64>)> {
    if m.qc.is_empty() {
        return None;
    }
    let mut t: Vec<(usize, usize, f64)> =
        m.qc.iter().map(|&(j, k, v)| if j == k { (j, j, 2.0 * v) } else { (j.min(k), j.max(k), v) }).collect();
    t.sort_by_key(|&(c, r, _)| (c, r));
    let mut e: Vec<(usize, usize, f64)> = Vec::with_capacity(t.len());
    for (c, r, x) in t {
        match e.last_mut() {
            Some(l) if (l.0, l.1) == (c, r) => l.2 += x,
            _ => e.push((c, r, x)),
        }
    }
    let mut p = vec![0 as Hi; m.n() + 1];
    for t in &e {
        p[t.0 + 1] += 1;
    }
    for c in 0..m.n() {
        p[c + 1] += p[c];
    }
    p.pop();
    let (i, v) = e.iter().map(|&(_, r, x)| (r as Hi, x)).unzip();
    Some((p, i, v))
}
