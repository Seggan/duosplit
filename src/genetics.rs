use crate::fitness::FitnessCalculator;
use crate::normal_distr::NormalDistribution;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use rand::rngs::ThreadRng;
use rand::{rng, Rng, RngExt};

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
            x: rng.random_range(-1.0..1.0),
        }
    }
}

pub fn j_k_from_i(i: f32, a: f32, c: f32, e: f32, b: f32, d: f32, f: f32) -> (f32, f32) {
    let denom = d * e - c * f;
    let j = (d + b * c * i - a * d * i) / denom;
    let k = (-f - b * e * i + a * f * i) / denom;
    (j, k)
}

pub struct Population {
    rng: ThreadRng,
    population: Vec<Genome>,
    fitnesses: Vec<f32>,
    elitism: usize,
    initial_std: f32,
    decay_rate: f32,
    generation: u32
}

impl Population {
    pub fn new_random(amount: usize, elitism: usize, initial_std: f32, decay_rate: f32) -> Self {
        let mut rng = rng();
        let mut population = Vec::with_capacity(amount);
        for _ in 0..amount {
            population.push(Genome::random(&mut rng));
        }
        Self {
            rng,
            population,
            fitnesses: Vec::new(),
            elitism,
            initial_std,
            decay_rate,
            generation: 0
        }
    }

    pub fn compute_generation(&mut self, calculator: &mut FitnessCalculator) -> Result<()> {
        let pop_size = self.population.len();
        self.fitnesses = calculator.compute_fitness(&self.population)?;
        let elite_indices = {
            let mut indices = (0..pop_size).collect::<Vec<usize>>();
            indices.sort_by(|&i, &j| self.fitnesses[i].partial_cmp(&self.fitnesses[j]).unwrap());
            indices[..self.elitism].to_vec()
        };
        let elites = elite_indices
            .iter()
            .map(|&i| self.population[i])
            .collect::<Vec<_>>();

        let mut new_population = elites.clone();
        let mutation_rate = self.initial_std * (-self.decay_rate * self.generation as f32).exp();
        while new_population.len() < pop_size {
            let idx1 = self.rng.random_range(0..pop_size);
            let mut idx2 = self.rng.random_range(0..pop_size);
            while idx2 == idx1 {
                idx2 = self.rng.random_range(0..pop_size);
            }
            let parent = if self.fitnesses[idx1] < self.fitnesses[idx2] {
                self.population[idx1]
            } else {
                self.population[idx2]
            };
            let child = Genome {
                i: parent.i + self.rng.sample(NormalDistribution::new(0.0, mutation_rate)),
                x: parent.x + self.rng.sample(NormalDistribution::new(0.0, mutation_rate)),
            };
            new_population.push(child);
        }

        self.population = new_population;
        self.generation += 1;

        Ok(())
    }

    pub fn best_genome(&self) -> (Genome, f32) {
        let (best_idx, _) = self.fitnesses
            .iter()
            .enumerate()
            .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        (self.population[best_idx], self.fitnesses[best_idx])
    }
}
