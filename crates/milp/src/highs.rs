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
    fn Highs_setDoubleOptionValue(h: *mut c_void, k: *const c_char, v: f64) -> Hi;
    fn Highs_run(h: *mut c_void) -> Hi;
    fn Highs_getModelStatus(h: *const c_void) -> Hi;
    fn Highs_getSolution(h: *const c_void, x: *mut f64, z: *mut f64, r: *mut f64, y: *mut f64) -> Hi;
    fn Highs_getDoubleInfoValue(h: *const c_void, k: *const c_char, v: *mut f64) -> Hi;
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
        h.flag("output_flag", false);
        h
    }

    pub fn flag(&self, k: &str, v: bool) {
        let k = CString::new(k).unwrap();
        // SAFETY: live handle, NUL-terminated key.
        unsafe { Highs_setBoolOptionValue(self.0, k.as_ptr(), v as Hi) };
    }

    pub fn real(&self, k: &str, v: f64) {
        let k = CString::new(k).unwrap();
        // SAFETY: live handle, NUL-terminated key.
        unsafe { Highs_setDoubleOptionValue(self.0, k.as_ptr(), v) };
    }

    /// Highs_passModel; every slice must have the length its argument count states.
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
    ) -> bool {
        // SAFETY: every array outlives the call and has the length HiGHS reads.
        let st = unsafe {
            Highs_passModel(
                self.0,
                c.len() as Hi,
                lo.len() as Hi,
                a.len() as Hi,
                0,
                ROWWISE,
                0,
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
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                integ.as_ptr(),
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
