// libtorch_cuda.so is force-loaded at runtime via dlopen() in pipeline.rs
// (the linker's --as-needed drops it otherwise).  This script only adds the
// search path so other torch-sys link-lib directives resolve correctly.
fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    let libtorch = match std::env::var("LIBTORCH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => return,
    };
    let lib_dir = libtorch.join("lib");
    if lib_dir.join("libtorch_cuda.so").exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:warning=libtorch_cuda.so found — will be loaded via dlopen() at runtime");
    } else {
        println!("cargo:warning=libtorch_cuda.so NOT found in {} — CPU only", lib_dir.display());
    }
}
