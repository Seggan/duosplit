use crate::bindings::Options;

mod bindings;

fn main() {
    let context = Box::leak(Box::new(bindings::Context::new_with_options(
        Options::new()
    ).unwrap()));
}