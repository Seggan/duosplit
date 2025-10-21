use std::f64::consts::PI;
use rand::distr::Distribution;
use rand::Rng;

pub struct NormalDistribution {
    mean: f64,
    std_dev: f64,
}

impl NormalDistribution {
    pub fn new(mean: f64, std_dev: f64) -> Self {
        NormalDistribution { mean, std_dev }
    }
}

impl Distribution<f64> for NormalDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        self.mean + z0 * self.std_dev
    }
}