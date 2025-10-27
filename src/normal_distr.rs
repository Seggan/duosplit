use std::f32::consts::PI;
use rand::distr::Distribution;
use rand::Rng;

pub struct NormalDistribution {
    mean: f32,
    std_dev: f32,
}

impl NormalDistribution {
    pub fn new(mean: f32, std_dev: f32) -> Self {
        NormalDistribution { mean, std_dev }
    }
}

impl Distribution<f32> for NormalDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        self.mean + z0 * self.std_dev
    }
}