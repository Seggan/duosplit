use bytemuck::{Pod, Zeroable};
use rand::Rng;

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct Genome {
    pub i: f32,
    pub x: f32,
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            i: rng.random_range(-1.0..1.0),
            x: rng.random_range(-1.0..1.0)
        }
    }
}

pub fn j_k_from_i(i: f32, a: f32, c: f32, e: f32, b: f32, d: f32, f: f32) -> (f32, f32) {
    let denom = d * e - c * f;
    let j = (d + b * c * i - a * d * i) / denom;
    let k = (-f - b * e * i + a * f * i) / denom;
    (j, k)
}