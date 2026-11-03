//! mmi: run the cases, or export a case's constraint matrix as a hypergraph.
//!
//!   mmi run FILE...   solve each case file, replay its solution and compare it with the oracle
//!                     (`Out::pass`); exits 1 if any case fails
//!   mmi hgr FILE      print the row-net hypergraph of the case's first model in hMETIS format:
//!                     vertices are columns, one hyperedge per row (Aykanat, Pinar and Çatalyürek,
//!                     SIAM J. Sci. Comput. 25, 2004); `tools/blocks.py` partitions it

use std::path::Path;
use std::process::ExitCode;

use hyper::Hg;

fn run(paths: &[String]) -> Result<bool, String> {
    let mut ok = true;
    for p in paths {
        let (case, data, oracle) = models::read(Path::new(p))?;
        let t = std::time::Instant::now();
        let o = models::run(&case, data).map_err(|e| format!("{p}: {e}"))?;
        let pass = o.pass(&oracle);
        ok &= pass;
        println!("== {p} [{case}] {:.2}s", t.elapsed().as_secs_f64());
        print!("{}", o.text);
        println!(
            "objective {:?} oracle {:?} violation {:.1e} -> {}",
            o.obj,
            oracle,
            o.viol,
            if pass { "pass" } else { "FAIL" }
        );
    }
    Ok(ok)
}

fn hgr(p: &str) -> Result<(), String> {
    let (case, data, _) = models::read(Path::new(p))?;
    let m = models::model(&case, data)?;
    print!("{}", Hg::new(m.n(), (0..m.m()).map(|i| m.cols(i).collect())).hgr());
    Ok(())
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let r = match a.first().map(String::as_str) {
        Some("run") if a.len() > 1 => run(&a[1..]),
        Some("hgr") if a.len() == 2 => hgr(&a[1]).map(|_| true),
        _ => Err("usage: mmi run FILE... | mmi hgr FILE".into()),
    };
    match r {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}
