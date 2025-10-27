use crate::cli::Cli;
use crate::genetics::{j_k_from_i, Genome};
use crate::gpu::{GpuContext, PixelUniform, QEUniform};
use crate::normal_distr::NormalDistribution;
use clap::Parser;
use fitrs::{Fits, FitsData, Hdu};
use ndarray::{s, Array2, Array3};
use rand::{rng, Rng};
use std::path::{Path, PathBuf};
use std::process::exit;

mod cli;
mod genetics;
mod gpu;
mod normal_distr;

#[pollster::main]
async fn main() {
    let cli = Cli::parse();

    println!("Reading FITS file: {}", cli.input.display());
    let (red_channel, green_channel, blue_channel) = match read_fits(&cli.input) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("Error reading FITS file: {}", err);
            exit(1);
        }
    };

    println!("Setting up GPU context...");
    let mut pixels = Vec::new();
    let flat_red = red_channel.flatten();
    let flat_green = green_channel.flatten();
    let flat_blue = blue_channel.flatten();
    for i in 0..flat_red.len() {
        pixels.push(PixelUniform {
            r: flat_red[i],
            g: flat_green[i],
            b: flat_blue[i],
        });
    }

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
    let context = GpuContext::new(pixels, (qe_red, qe_green, qe_blue)).await;

    println!("Starting genetic algorithm optimization...");
    let best_genome = optimized_genome(&cli, context).await;

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
    let h_alpha = ha_r * &red_channel + ha_g * &green_channel + ha_b * &blue_channel;

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
    let oiii = oiii_r * &red_channel + oiii_g * &green_channel + oiii_b * &blue_channel;

    println!("Best genome results:");
    println!(
        "H-alpha coefficients: r = {}, g = {}, b = {}",
        ha_r, ha_g, ha_b
    );
    println!(
        "OIII coefficients: r = {}, g = {}, b = {}",
        oiii_r, oiii_g, oiii_b
    );

    if let Err(err) = write_fits(&cli.output.join("h_alpha.fit"), &h_alpha) {
        eprintln!("Error writing H-alpha FITS file: {}", err);
        exit(1);
    }

    if let Err(err) = write_fits(&cli.output.join("oiii.fit"), &oiii) {
        eprintln!("Error writing OIII FITS file: {}", err);
        exit(1);
    }

    println!("Done!");
}

fn read_fits(path: &impl AsRef<Path>) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), String> {
    let image = Fits::open(path).map_err(|e| format!("Failed to open FITS file: {}", e))?;
    let hdu = image.get(0).ok_or("No HDU found in FITS file")?;
    let (shape, data) = match hdu.read_data() {
        FitsData::Characters(arr) => (
            arr.shape,
            arr.data.into_iter().map(|v| v as u64 as f32).collect(),
        ),
        FitsData::IntegersI32(arr) => (
            arr.shape,
            arr.data
                .into_iter()
                .map(|v| v.unwrap_or(0) as f32)
                .collect(),
        ),
        FitsData::IntegersU32(arr) => (
            arr.shape,
            arr.data
                .into_iter()
                .map(|v| v.unwrap_or(0) as f32)
                .collect(),
        ),
        FitsData::FloatingPoint32(arr) => (arr.shape, arr.data),
        FitsData::FloatingPoint64(arr) => {
            eprintln!(
                "Warning: Converting FITS data from 64 bit to 32 bit; this may lose precision."
            );
            (arr.shape, arr.data.into_iter().map(|v| v as f32).collect())
        }
    };

    let channels = Array3::from_shape_vec((shape[2], shape[1], shape[0]), data)
        .expect("Failed to reshape FITS data into 3D array");
    let red_channel = channels.slice(s![0, .., ..]).into_owned();
    let green_channel = channels.slice(s![1, .., ..]).into_owned();
    let blue_channel = channels.slice(s![2, .., ..]).into_owned();
    Ok((red_channel, green_channel, blue_channel))
}

async fn optimized_genome(cli: &Cli, context: GpuContext) -> Genome {
    let mut rng = rng();
    let mut population = Vec::with_capacity(cli.population_size);
    for _ in 0..cli.population_size {
        population.push(Genome::random(&mut rng));
    }

    let mut fitnesses = Vec::new();
    for gen in 0..cli.generations {
        fitnesses = context.compute_fitness(&population).await;

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
    }

    let (best_genome, best_fitness) = best_genome_and_fitness(&population, &fitnesses);
    println!("Best genome found with noise: {}", best_fitness);
    if best_genome.i < best_genome.x {
        println!("Warning: H-alpha component is less than OIII component; they may be swapped.");
        Genome {
            i: best_genome.x,
            x: best_genome.i,
        }
    } else {
        best_genome
    }
}

fn best_genome_and_fitness(population: &Vec<Genome>, fitnesses: &Vec<f32>) -> (Genome, f32) {
    let (best_idx, _) = fitnesses
        .iter()
        .enumerate()
        .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (population[best_idx], fitnesses[best_idx])
}

fn write_fits(path: &PathBuf, data: &Array2<f32>) -> Result<(), String> {
    let hdu = Hdu::new(
        &[data.shape()[1], data.shape()[0]],
        data.as_slice().unwrap().to_vec(),
    );
    Fits::create(path, hdu)
        .map(|_| ())
        .map_err(|e| format!("Failed to write to {}: {}", path.to_str().unwrap(), e))
}
