// torch-sys (tch 0.18) does not always auto-link libtorch_cuda.so on Linux
// when TORCH_CUDA_VERSION is absent.  This build script detects the .so
// directly and adds the explicit -ltorch_cuda link flag so the GPU code
// path in libtorch_cpu.so can find and dispatch into the CUDA backend.
fn main() {
    // Re-run this script whenever LIBTORCH changes.
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    let libtorch = match std::env::var("LIBTORCH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => return,
    };
    let lib_dir = libtorch.join("lib");
    let cuda_so = lib_dir.join("libtorch_cuda.so");
    if cuda_so.exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=torch_cuda");
        println!("cargo:warning=libtorch_cuda.so found — linking CUDA backend");
    } else {
        println!("cargo:warning=libtorch_cuda.so NOT found in {} — CPU only", lib_dir.display());
    }
}
