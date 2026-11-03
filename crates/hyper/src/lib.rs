//! Weighted hypergraphs G = (V, E, w) in the convention of DeepHypergraph (iMoonLab/DeepHypergraph,
//! `dhg.Hypergraph(num_v, e_list, e_weight)`), with the operator of Zhou, Huang and Schölkopf,
//! "Learning with hypergraphs: clustering, classification, and embedding", NIPS 2006.
//!
//! Legend
//!   V = {0..n}  vertices                 e ⊆ V  hyperedge, stored sorted without repeats
//!   H ......... incidence, H(v, e) = 1 iff v ∈ e        w(e) ... hyperedge weight, W = diag(w)
//!   d(v) ...... Σ_{e ∋ v} w(e), Dv = diag(d)            δ(e) ... |e|, De = diag(δ)
//!   Θ ......... Dv^{-1/2} H W De^{-1} Hᵀ Dv^{-1/2} (DHG `L_HGNN`); Δ = I − Θ is Zhou's Laplacian (DHG `L_sym`)

/// A hypergraph on n vertices. Duplicate hyperedges are kept, so they act as one edge of summed
/// weight in Θ (DHG merges them by `merge_op`, default the mean weight).
#[derive(Clone, Debug)]
pub struct Hg {
    pub n: usize,
    pub e: Vec<Vec<usize>>,
    pub w: Vec<f64>,
}

impl Hg {
    /// G with unit weights; each hyperedge is sorted and its repeated vertices removed.
    pub fn new(n: usize, e: impl IntoIterator<Item = Vec<usize>>) -> Hg {
        let e: Vec<Vec<usize>> = e
            .into_iter()
            .map(|mut x| {
                x.sort_unstable();
                x.dedup();
                assert!(x.last().is_none_or(|&v| v < n), "hyperedge vertex out of range 0..{n}");
                x
            })
            .collect();
        let w = vec![1.0; e.len()];
        Hg { n, e, w }
    }

    /// d(v) for every vertex.
    pub fn d(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.n];
        for (e, &w) in self.e.iter().zip(&self.w) {
            for &v in e {
                d[v] += w;
            }
        }
        d
    }

    /// The hyperedges at each vertex, {e : v ∈ e} (DHG `nbr_e`), in edge order.
    pub fn nbr_e(&self) -> Vec<Vec<usize>> {
        let mut s = vec![Vec::new(); self.n];
        for (k, e) in self.e.iter().enumerate() {
            for &v in e {
                s[v].push(k);
            }
        }
        s
    }

    /// Θ f, in O(Σ_e δ(e)). Isolated vertices (d = 0) map to 0.
    pub fn theta(&self, f: &[f64]) -> Vec<f64> {
        let s: Vec<f64> = self.d().iter().map(|&d| if d > 0.0 { d.powf(-0.5) } else { 0.0 }).collect();
        let mut g = vec![0.0; self.n];
        for (e, &w) in self.e.iter().zip(&self.w) {
            if e.is_empty() {
                continue;
            }
            let t = w / e.len() as f64 * e.iter().map(|&v| s[v] * f[v]).sum::<f64>();
            for &v in e {
                g[v] += s[v] * t;
            }
        }
        g
    }

    /// Zhou et al. transductive ranking: f = (1 − α)(I − αΘ)⁻¹ y, 0 < α < 1, computed by
    /// f ← αΘf + (1 − α)y from f = y. Θ has spectral radius ≤ 1, so the error contracts by α per
    /// step; iteration stops when no entry moves by more than 1e-12 (‖y‖∞ scale) or after 10⁴ steps.
    pub fn rank(&self, y: &[f64], alpha: f64) -> Vec<f64> {
        assert!(0.0 < alpha && alpha < 1.0, "alpha must lie in (0, 1)");
        let tol = 1e-12 * y.iter().fold(0.0, |a: f64, &b| a.max(b.abs())).max(1.0);
        let mut f = y.to_vec();
        for _ in 0..10_000 {
            let g: Vec<f64> = self.theta(&f).iter().zip(y).map(|(t, &yv)| alpha * t + (1.0 - alpha) * yv).collect();
            let r = g.iter().zip(&f).fold(0.0, |a: f64, (p, q)| a.max((p - q).abs()));
            f = g;
            if r <= tol {
                break;
            }
        }
        f
    }

    /// The hMETIS hypergraph file (the input format of KaHyPar and Mt-KaHyPar): a header
    /// "|E| |V|", then one line per hyperedge listing its vertices 1-based. Unit weights only.
    pub fn hgr(&self) -> String {
        let mut s = format!("{} {}\n", self.e.len(), self.n);
        for e in &self.e {
            let l: Vec<String> = e.iter().map(|v| (v + 1).to_string()).collect();
            s.push_str(&l.join(" "));
            s.push('\n');
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g() -> Hg {
        // V = {0, 1, 2, 3}, E = {{0, 1, 2}, {2, 3}}; d = (1, 1, 2, 1).
        Hg::new(4, [vec![2, 0, 1, 0], vec![3, 2]])
    }

    #[test]
    fn theta_matches_the_dense_product() {
        let g = g();
        assert_eq!(g.e[0], vec![0, 1, 2]);
        // Θ(u, v) = Σ_{e ∋ u, v} w(e) / (δ(e) √(d(u) d(v))), from the definition.
        let d = g.d();
        let f = [1.0, -2.0, 0.5, 3.0];
        let t = g.theta(&f);
        for u in 0..4 {
            let mut x = 0.0;
            for v in 0..4 {
                for e in &g.e {
                    if e.contains(&u) && e.contains(&v) {
                        x += f[v] / (e.len() as f64 * (d[u] * d[v]).sqrt());
                    }
                }
            }
            assert!((t[u] - x).abs() < 1e-12);
        }
    }

    #[test]
    fn rank_solves_the_linear_system() {
        let g = g();
        let y = [1.0, 0.0, 0.0, 0.0];
        let f = g.rank(&y, 0.9);
        let t = g.theta(&f);
        for v in 0..4 {
            assert!((f[v] - 0.9 * t[v] - 0.1 * y[v]).abs() < 1e-10);
        }
        // Mass spreads within the labelled vertex's edge first: f(1) > f(3).
        assert!(f[1] > f[3] && f[3] > 0.0);
    }

    #[test]
    fn nbr_e_inverts_the_incidence() {
        assert_eq!(g().nbr_e(), vec![vec![0], vec![0], vec![0, 1], vec![1]]);
    }

    #[test]
    fn hgr_is_one_based() {
        assert_eq!(g().hgr(), "2 4\n1 2 3\n3 4\n");
    }
}
