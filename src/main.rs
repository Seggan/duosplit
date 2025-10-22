use crate::bindings::{ArrayF64D1, Context, Options, QE};
use crate::cli::Cli;
use crate::genetics::Genome;
use crate::normal_distr::NormalDistribution;
use clap::Parser;
use fitrs::{Fits, FitsData, Hdu};
use ndarray::{s, Array2, Array3};
use rand::{rng, Rng};
use std::path::{Path, PathBuf};
use std::process::exit;

mod bindings;
mod cli;
mod genetics;
mod normal_distr;

const POP_SIZE: usize = 100;
const GENS: u32 = 250;
const INITIAL_STD: f64 = 0.5;
const DECAY_RATE: f64 = 0.1;
const ELITISM: usize = 5;

fn main() {
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
    let context = Context::new_with_options(Options::new()).unwrap();
    let gpu_qe_red = QE::new(&context, cli.red_ha_qe, cli.red_oiii_qe).unwrap();
    let gpu_qe_green = QE::new(&context, cli.green_ha_qe, cli.green_oiii_qe).unwrap();
    let gpu_qe_blue = QE::new(&context, cli.blue_ha_qe, cli.blue_oiii_qe).unwrap();

    println!("Starting genetic algorithm optimization...");
    let best_genome = optimized_genome(
        &context,
        &red_channel,
        &green_channel,
        &blue_channel,
        gpu_qe_red,
        gpu_qe_green,
        gpu_qe_blue,
    );

    let ha_r = best_genome.i;
    let (ha_g, ha_b) = context
        .j_k_from_i(
            ha_r,
            cli.red_ha_qe,
            cli.green_ha_qe,
            cli.blue_ha_qe,
            cli.red_oiii_qe,
            cli.green_oiii_qe,
            cli.blue_oiii_qe,
        )
        .unwrap();
    let h_alpha = ha_r * &red_channel + ha_g * &green_channel + ha_b * &blue_channel;

    let oiii_r = best_genome.x;
    let (oiii_g, oiii_b) = context
        .j_k_from_i(
            oiii_r,
            cli.red_oiii_qe,
            cli.green_oiii_qe,
            cli.blue_oiii_qe,
            cli.red_ha_qe,
            cli.green_ha_qe,
            cli.blue_ha_qe,
        )
        .unwrap();
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

fn read_fits(path: &impl AsRef<Path>) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), String> {
    let image = Fits::open(path).map_err(|e| format!("Failed to open FITS file: {}", e))?;
    let hdu = image.get(0).ok_or("No HDU found in FITS file")?;
    let (shape, data) = match hdu.read_data() {
        FitsData::Characters(arr) => (
            arr.shape,
            arr.data.into_iter().map(|v| v as u64 as f64).collect(),
        ),
        FitsData::IntegersI32(arr) => (
            arr.shape,
            arr.data
                .into_iter()
                .map(|v| v.unwrap_or(0) as f64)
                .collect(),
        ),
        FitsData::IntegersU32(arr) => (
            arr.shape,
            arr.data
                .into_iter()
                .map(|v| v.unwrap_or(0) as f64)
                .collect(),
        ),
        FitsData::FloatingPoint32(arr) => {
            (arr.shape, arr.data.into_iter().map(|v| v as f64).collect())
        }
        FitsData::FloatingPoint64(arr) => (arr.shape, arr.data),
    };

    let channels = Array3::from_shape_vec((shape[2], shape[1], shape[0]), data)
        .expect("Failed to reshape FITS data into 3D array");
    let red_channel = channels.slice(s![0, .., ..]).into_owned();
    let green_channel = channels.slice(s![1, .., ..]).into_owned();
    let blue_channel = channels.slice(s![2, .., ..]).into_owned();
    Ok((red_channel, green_channel, blue_channel))
}

fn optimized_genome(
    context: &Context,
    red: &Array2<f64>,
    green: &Array2<f64>,
    blue: &Array2<f64>,
    qe_red: QE,
    qe_green: QE,
    qe_blue: QE,
) -> Genome {
    let image = context
        .new_image(
            &ArrayF64D1::new(
                &context,
                [red.len() as i64],
                red.flatten().as_slice().unwrap(),
            )
            .unwrap(),
            &ArrayF64D1::new(
                &context,
                [green.len() as i64],
                green.flatten().as_slice().unwrap(),
            )
            .unwrap(),
            &ArrayF64D1::new(
                &context,
                [blue.len() as i64],
                blue.flatten().as_slice().unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

    let mut rng = rng();
    let mut population = Vec::with_capacity(POP_SIZE);
    for _ in 0..POP_SIZE {
        population.push(Genome::random(&mut rng));
    }

    let mut fitnesses = vec![0.0; POP_SIZE];
    for gen in 0..GENS {
        fitnesses = context
            .fitness(
                &Genome::list_as_gpu_list(&context, &population),
                &image,
                &qe_red,
                &qe_green,
                &qe_blue,
            )
            .unwrap()
            .get()
            .unwrap();

        let elite_indices = {
            let mut indices = (0..POP_SIZE).collect::<Vec<usize>>();
            indices.sort_by(|&i, &j| fitnesses[i].partial_cmp(&fitnesses[j]).unwrap());
            indices[..ELITISM].to_vec()
        };
        let elites = elite_indices
            .iter()
            .map(|&i| population[i])
            .collect::<Vec<Genome>>();

        let mut new_population = elites.clone();
        let mutation_rate = INITIAL_STD * (-DECAY_RATE * gen as f64).exp();
        while new_population.len() < POP_SIZE {
            let idx1 = rng.random_range(0..POP_SIZE);
            let mut idx2 = rng.random_range(0..POP_SIZE);
            while idx2 == idx1 {
                idx2 = rng.random_range(0..POP_SIZE);
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

fn best_genome_and_fitness(population: &Vec<Genome>, fitnesses: &Vec<f64>) -> (Genome, f64) {
    let (best_idx, _) = fitnesses
        .iter()
        .enumerate()
        .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (population[best_idx], fitnesses[best_idx])
}

fn write_fits(path: &PathBuf, data: &Array2<f64>) -> Result<(), String> {
    let hdu = Hdu::new(
        &[data.shape()[1], data.shape()[0]],
        data.as_slice().unwrap().to_vec(),
    );
    Fits::create(path, hdu)
        .map(|_| ())
        .map_err(|e| format!("Failed to write to {}: {}", path.to_str().unwrap(), e))
}
