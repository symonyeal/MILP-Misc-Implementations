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
}
