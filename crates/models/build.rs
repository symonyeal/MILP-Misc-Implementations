//! Gives `mmi` an rpath to the prebuilt SCIP that scip-sys downloaded, so the binary runs when
//! started directly, not only under `cargo run`. Windows has no rpath; milp's build script copies
//! the DLLs beside the binaries instead.

fn main() {
    let t = std::env::var("TARGET").unwrap();
    if let (false, Ok(lib)) = (t.contains("windows"), std::env::var("DEP_HIGHS_SCIP_LIBDIR")) {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{lib}");
    }
}
