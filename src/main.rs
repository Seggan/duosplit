use crate::cli::Cli;
use crate::fitness::{FitnessCalculator, QEUniform};
use crate::fits::write_fits;
use crate::genetics::{j_k_from_i, Genome};
use crate::gpu::GpuDevice;
use crate::normal_distr::NormalDistribution;
use anyhow::Result;
use clap::Parser;
use ndarray::s;
use rand::{rng, Rng};
use std::process::exit;
use std::time::Instant;

mod cli;
mod genetics;
mod normal_distr;
mod gpu;
mod fitness;
mod fits;

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Reading FITS file: {}", cli.input.display());
    let image = match fits::read_fits(&cli.input) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("Error reading FITS file: {}", err);
            exit(1);
        }
    };

    println!("Setting up GPU device...");
    let device = GpuDevice::new()?;
    println!("Using GPU: {}", device.get_gpu_name());

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
    let context = FitnessCalculator::new(&device, &image, cli.chunks, (qe_red, qe_green, qe_blue))?;

    println!("Starting genetic algorithm optimization...");
    let best_genome = optimized_genome(&cli, context)?;

    let ha_r = best_genome.i;
    let (ha_g, ha_b) = j_k_from_i(
        ha_r,
        cli.red_ha_qe,
        cli.green_ha_qe,
        cli.blue_ha_qe,
        cli.red_oiii_qe,
        cli.green_oiii_qe,
        cli.blue_oiii_qe,
    );

    let oiii_r = best_genome.x;
    let (oiii_g, oiii_b) = j_k_from_i(
        oiii_r,
        cli.red_oiii_qe,
        cli.green_oiii_qe,
        cli.blue_oiii_qe,
        cli.red_ha_qe,
        cli.green_ha_qe,
        cli.blue_ha_qe,
    );

    println!("Best genome results:");
    println!(
        "H-alpha coefficients: r = {}, g = {}, b = {}",
        ha_r, ha_g, ha_b
    );
    println!(
        "OIII coefficients: r = {}, g = {}, b = {}",
        oiii_r, oiii_g, oiii_b
    );

    let red_channel = image.slice(s![.., .., 0]);
    let green_channel = image.slice(s![.., .., 1]);
    let blue_channel = image.slice(s![.., .., 2]);
    let h_alpha = ha_r * &red_channel + ha_g * &green_channel + ha_b * &blue_channel;
    let mut oiii = oiii_r * &red_channel + oiii_g * &green_channel + oiii_b * &blue_channel;

    if cli.normalize {
        let mean_ha = h_alpha.mean().unwrap();
        let mean_oiii = oiii.mean().unwrap();
        oiii = oiii - mean_oiii + mean_ha;
    }

    write_fits(&cli.output.join("h_alpha.fit"), h_alpha)?;
    write_fits(&cli.output.join("oiii.fit"), oiii)?;

    println!("Done!");

    Ok(())
}

fn optimized_genome(cli: &Cli, mut context: FitnessCalculator) -> Result<Genome> {
    let mut rng = rng();
    let mut population = Vec::with_capacity(cli.population_size);
    for _ in 0..cli.population_size {
        population.push(Genome::random(&mut rng));
    }

    let mut fitnesses = Vec::new();
    for gen in 0..cli.generations {
        let start = Instant::now();
        fitnesses = context.compute_fitness(&population)?;

        let elite_indices = {
            let mut indices = (0..cli.population_size).collect::<Vec<usize>>();
            indices.sort_by(|&i, &j| fitnesses[i].partial_cmp(&fitnesses[j]).unwrap());
            indices[..cli.elitism].to_vec()
        };
        let elites = elite_indices
            .iter()
            .map(|&i| population[i])
            .collect::<Vec<Genome>>();

        let mut new_population = elites.clone();
        let mutation_rate = cli.initial_std * (-cli.decay_rate * gen as f32).exp();
        while new_population.len() < cli.population_size {
            let idx1 = rng.random_range(0..cli.population_size);
            let mut idx2 = rng.random_range(0..cli.population_size);
            while idx2 == idx1 {
                idx2 = rng.random_range(0..cli.population_size);
            }
            let parent = if fitnesses[idx1] < fitnesses[idx2] {
                population[idx1]
            } else {
                population[idx2]
            };
            let child = Genome {
                i: parent.i + rng.sample(NormalDistribution::new(0.0, mutation_rate)),
                x: parent.x + rng.sample(NormalDistribution::new(0.0, mutation_rate)),
            };
            new_population.push(child);
        }

        population = new_population;
        let (_, best_fitness) = best_genome_and_fitness(&population, &fitnesses);
        println!("Generation {}: {}", gen, best_fitness);
        if cli.timings {
            let duration = Instant::now() - start;
            println!("Generation {} took {:?}", gen, duration);
        }
    }

    let (best_genome, best_fitness) = best_genome_and_fitness(&population, &fitnesses);
    println!("Best genome found with noise: {}", best_fitness);
    Ok(if best_genome.i < best_genome.x {
        println!("Warning: H-alpha component is less than OIII component; they may be swapped.");
        Genome {
            i: best_genome.x,
            x: best_genome.i,
        }
    } else {
        best_genome
    })
}

fn best_genome_and_fitness(population: &Vec<Genome>, fitnesses: &Vec<f32>) -> (Genome, f32) {
    let (best_idx, _) = fitnesses
        .iter()
        .enumerate()
        .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (population[best_idx], fitnesses[best_idx])
}