//! Builds the HiGHS submodule (tag v1.12.0) as a static library.

use std::{env, path::Path};

fn main() {
    let src = Path::new("HiGHS");
    if !src.join("highs").exists() {
        panic!("HiGHS source missing: run `git submodule update --init` (crates/milp/HiGHS)");
    }
    let crt = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let crt = if crt.split(',').any(|f| f == "crt-static") { "MultiThreaded" } else { "MultiThreadedDLL" };
    // The cmake crate replaces CMAKE_<LANG>_FLAGS_RELEASE with cc's flags, which carry no
    // optimisation level: without these HiGHS builds unoptimised with asserts on (about 5x slower).
    let t = env::var("TARGET").unwrap();
    let opt = if t.contains("msvc") { "/O2 /DNDEBUG" } else { "-O2 -DNDEBUG -ffp-contract=off" };
    let dst = cmake::Config::new(src)
        .profile("Release")
        .cflag(opt)
        .cxxflag(opt)
        .define("FAST_BUILD", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_CXX_EXE", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("ZLIB", "OFF")
        .define("CMAKE_MSVC_RUNTIME_LIBRARY", crt)
        .define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "FALSE")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=highs");
    if t.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if t.contains("linux") || t.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=stdc++");
    }
    println!("cargo:rerun-if-changed=HiGHS/Version.txt");
    println!("cargo:rerun-if-changed=build.rs");
    if t.contains("windows") {
        copy_scip_dlls();
    }
}

/// Windows has no rpath: put the DLLs of scip-sys's prebuilt SCIP beside the binaries in the
/// profile directory (target/<profile>), where the loader looks first. `cargo test` and `cargo
/// run` also find them through the link-search path, but a binary started directly needs this.
fn copy_scip_dlls() {
    let lib = env::var("DEP_SCIP_LIBDIR").expect("scip-sys did not report its library directory");
    let bin = Path::new(&lib).join("../bin");
    let out = env::var("OUT_DIR").unwrap();
    let prof = Path::new(&out).ancestors().nth(3).unwrap();
    for e in std::fs::read_dir(&bin).unwrap_or_else(|e| panic!("{}: {e}", bin.display())) {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("dll")) {
            let to = prof.join(p.file_name().unwrap());
            if !to.exists() {
                std::fs::copy(&p, &to).unwrap();
            }
        }
    }
}
