use crate::cli::Cli;
use crate::fitness::{FitnessCalculator, QEUniform};
use crate::fits::write_fits;
use crate::genetics::{j_k_from_i, Genome, Population};
use crate::gpu::GpuDevice;
use anyhow::Result;
use clap::Parser;
use ndarray::s;
use std::process::exit;
use std::time::Instant;

mod cli;
mod fitness;
mod fits;
mod genetics;
mod gpu;
mod normal_distr;

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
    let mut population = Population::new_random(
        cli.population_size,
        cli.elitism,
        cli.initial_std,
        cli.decay_rate,
    );

    for gen in 0..cli.generations {
        let start = Instant::now();
        population.compute_generation(&mut context)?;
        let (_, best_fitness) = population.best_genome();
        println!("Generation {}: {}", gen, best_fitness);
        if cli.timings {
            let duration = Instant::now() - start;
            println!("Generation {} took {:?}", gen, duration);
        }
    }

    let (best_genome, best_fitness) = population.best_genome();
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