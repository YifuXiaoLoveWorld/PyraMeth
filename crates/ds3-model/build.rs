// torch-sys emits cargo:rustc-link-lib=torch_cuda, but the linker's --as-needed
// drops it because no Rust symbol directly references it.  libtorch_cuda.so's
// static initializers must run to register PyTorch's CUDA dispatch backend;
// without them tch::Cuda::device_count() returns 0.
//
// Fix: wrap the library in --push-state/--pop-state so --no-as-needed applies
// only to this one .so, leaving everything else unchanged.
fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    let libtorch = match std::env::var("LIBTORCH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => return,
    };
    let lib_dir = libtorch.join("lib");
    let cuda_so = lib_dir.join("libtorch_cuda.so");
    if cuda_so.exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        // Pass the absolute path directly inside a push/pop-state block.
        // Direct-path form bypasses -l naming rules; push/pop-state scopes
        // --no-as-needed so the linker cannot discard this library.
        println!("cargo:rustc-link-arg=-Wl,--push-state,--no-as-needed");
        println!("cargo:rustc-link-arg={}", cuda_so.display());
        println!("cargo:rustc-link-arg=-Wl,--pop-state");
        println!("cargo:warning=libtorch_cuda.so: forced inclusion via --no-as-needed");
    } else {
        println!("cargo:warning=libtorch_cuda.so NOT found in {} — CPU only", lib_dir.display());
    }
}
