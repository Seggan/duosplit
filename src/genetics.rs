use crate::cli::Cli;
use crate::fitness::{FitnessCalculator, QEUniform};
use crate::gpu::GpuDevice;
use crate::normal_distr::NormalDistribution;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use ndarray::{s, Array2, Array3};
use rand::rngs::ThreadRng;
use rand::{rng, Rng, RngExt};
use std::cmp::Ordering;
use std::time::Instant;

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
    generation: u32,
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
            generation: 0,
        }
    }

    pub fn compute_generation(
        &mut self,
        calculator: &mut FitnessCalculator,
        show_timings: bool,
    ) -> Result<()> {
        let start = Instant::now();

        let pop_size = self.population.len();
        self.fitnesses = calculator.compute_fitness(&self.population)?;
        let elite_indices = {
            let mut indices = (0..pop_size).collect::<Vec<usize>>();
            indices.sort_by(|&i, &j| self.fitnesses[i].partial_cmp(&self.fitnesses[j]).unwrap_or(Ordering::Greater));
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

        println!("Generation {}: {}", self.generation, self.best_genome().1);
        if show_timings {
            let duration = Instant::now() - start;
            println!("Generation {} took {:?}", self.generation, duration);
        }

        Ok(())
    }

    pub fn best_genome(&self) -> (Genome, f32) {
        let (best_idx, _) = self
            .fitnesses
            .iter()
            .enumerate()
            .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let mut genome = self.population[best_idx];
        if genome.i < genome.x {
            println!("Warning: H-alpha component is less than OIII component; they may be swapped.");
            genome = Genome {
                i: genome.x,
                x: genome.i,
            }
        }
        (genome, self.fitnesses[best_idx])
    }
}

pub struct TileProcessor {
    generations: u32,
    qes: (QEUniform, QEUniform, QEUniform),
    tile: Array3<f32>,
    calculator: FitnessCalculator,
    population: Population,
    x_start: usize,
    x_end: usize,
    y_start: usize,
    y_end: usize,
}

impl TileProcessor {
    pub fn gen_tiles(
        cli: &Cli,
        gpu: &GpuDevice,
        image: &Array3<f32>,
    ) -> Result<Vec<TileProcessor>> {
        let qe_red = QEUniform {
            ha: cli.red_ha_qe,
            oiii: cli.red_oiii_qe,
        };
        let qe_green = QEUniform {
            ha: cli.green_ha_qe,
            oiii: cli.green_oiii_qe,
        };
        let qe_blue = QEUniform {
            ha: cli.blue_ha_qe,
            oiii: cli.blue_oiii_qe,
        };

        let mut tiles = Vec::new();
        let image_width = image.shape()[0];
        let image_height = image.shape()[1];
        let (width, height) = determine_optimal_tile(cli.tiles, image_width, image_height);
        let rows = image_height / height;
        let columns = image_width / width;
        for y in 0..rows {
            let y_start = y * height;
            let y_end = if y == rows - 1 {
                image_height
            } else {
                y_start + height
            };
            for x in 0..columns {
                let x_start = x * width;
                let x_end = if x == columns - 1 {
                    image_width
                } else {
                    x_start + width
                };

                let tile = image
                    .slice(s![x_start..x_end, y_start..y_end, ..])
                    .as_standard_layout()
                    .into_owned();
                let calculator =
                    FitnessCalculator::new(gpu, &tile, cli.chunks, (qe_red, qe_green, qe_blue))?;
                let population = Population::new_random(
                    cli.population_size,
                    cli.elitism,
                    cli.initial_std,
                    cli.decay_rate,
                );
                let processor = TileProcessor {
                    generations: cli.generations,
                    qes: (qe_red, qe_green, qe_blue),
                    tile,
                    calculator,
                    population,
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                };
                tiles.push(processor);
            }
        }
        Ok(tiles)
    }

    pub fn run_genetic_algorithm(&mut self, show_timings: bool) -> Result<()> {
        for _ in 0..self.generations {
            self.population
                .compute_generation(&mut self.calculator, show_timings)?;
        }
        Ok(())
    }

    pub fn output(&mut self, ha: &mut Array2<f32>, oiii: &mut Array2<f32>) {
        let r = self.tile.slice(s![.., .., 0]);
        let g = self.tile.slice(s![.., .., 1]);
        let b = self.tile.slice(s![.., .., 2]);

        let (best_genome, _) = self.population.best_genome();
        let ha_r = best_genome.i;
        let (ha_g, ha_b) = j_k_from_i(
            ha_r,
            self.qes.0.ha,
            self.qes.1.ha,
            self.qes.2.ha,
            self.qes.0.oiii,
            self.qes.1.oiii,
            self.qes.2.oiii,
        );
        ha.slice_mut(s![self.x_start..self.x_end, self.y_start..self.y_end])
            .assign(&(ha_r * &r + ha_g * &g + ha_b * &b));

        let oiii_r = best_genome.x;
        let (oiii_g, oiii_b) = j_k_from_i(
            oiii_r,
            self.qes.0.oiii,
            self.qes.1.oiii,
            self.qes.2.oiii,
            self.qes.0.ha,
            self.qes.1.ha,
            self.qes.2.ha,
        );
        oiii.slice_mut(s![self.x_start..self.x_end, self.y_start..self.y_end])
            .assign(&(oiii_r * &r + oiii_g * &g + oiii_b * &b));
    }
}

fn determine_optimal_tile(x: usize, w: usize, h: usize) -> (usize, usize) {
    let mut best_error = usize::MAX;
    let mut best = (0, 0);
    for r in 1..=x.isqrt() {
        if x % r != 0 {
            continue;
        }

        let c = x / r;

        let a = (w + c / 2) / c;
        let b = (h + r / 2) / r;

        let error = a.abs_diff(b);

        if error < best_error {
            best = (a, b);
            best_error = error;
        }
    }
    best
}
