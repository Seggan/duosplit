use futhark_bindgen::Backend;

fn main() {
    futhark_bindgen::build(Backend::OpenCL, "./src/futhark/alg.fut", "bindings.rs");
}