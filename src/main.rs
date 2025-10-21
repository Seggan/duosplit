use crate::bindings::{ArrayF64D1, Context, Options};
use crate::config::Camera;
use crate::genetics::Genome;
use crate::normal_distr::NormalDistribution;
use fitrs::{Fits, FitsData, Hdu};
use ndarray::{s, Array2, Array3, CowArray, Ix1};
use rand::{rng, Rng};
use std::fs::File;
use std::path::Path;

mod bindings;
mod config;
mod genetics;
mod normal_distr;

const POP_SIZE: usize = 100;
const GENS: u32 = 500;
const INITIAL_STD: f64 = 0.1;
const DECAY_RATE: f64 = 0.01;
const ELITISM: usize = 50;
const LAMBDA: f64 = 1e2;

fn main() {
    let config = Path::new("config.json");
    let cameras = serde_json::from_reader::<_, Vec<Camera>>(
        File::open(config).expect("Failed to open config file"),
    )
    .expect("Failed to parse config file");
    let camera = &cameras[0];
    println!("Using camera: {:?}", camera);

    let image = Fits::open("image.fit").expect("Failed to open FITS file");
    let (shape, data) = match image.get(0).expect("No HDU found").read_data() {
        FitsData::Characters(_) => panic!("Did not expect character data"),
        FitsData::IntegersI32(_) => panic!("Did not expect i32 data"),
        FitsData::IntegersU32(_) => panic!("Did not expect u32 data"),
        FitsData::FloatingPoint32(arr) => (
            arr.shape,
            arr.data.into_iter().map(|v| v as f64).collect::<Vec<f64>>(),
        ),
        FitsData::FloatingPoint64(arr) => (arr.shape, arr.data),
    };
    let channels = Array3::from_shape_vec((shape[2], shape[1], shape[0]), data).unwrap();

    let red_channel = channels.slice(s![0, .., ..]).into_owned();
    let green_channel = channels.slice(s![1, .., ..]).into_owned();
    let blue_channel = channels.slice(s![2, .., ..]).into_owned();

    let mut best_genome = optimized_genome(camera, &red_channel, &green_channel, &blue_channel);
    if best_genome.ha_r < best_genome.oiii_r {
        // H-alpha was mistaken for OIII; swap them
        best_genome = Genome {
            ha_r: best_genome.oiii_r,
            ha_g: best_genome.oiii_g,
            ha_b: best_genome.oiii_b,
            oiii_r: best_genome.ha_r,
            oiii_g: best_genome.ha_g,
            oiii_b: best_genome.ha_b,
        };
        println!("H-alpha and OIII components were mixed up; swapped them.");
    }

    let h_alpha = best_genome.ha_r * &red_channel
        + best_genome.ha_g * &green_channel
        + best_genome.ha_b * &blue_channel;
    let oiii = best_genome.oiii_r * &red_channel
        + best_genome.oiii_g * &green_channel
        + best_genome.oiii_b * &blue_channel;

    let ha_hdu = Hdu::new(
        &[h_alpha.shape()[1], h_alpha.shape()[0]],
        h_alpha.as_slice().unwrap().to_vec()
    );
    Fits::create("h_alpha.fit", ha_hdu).expect("Failed to write H-alpha FITS file");

    let oiii_hdu = Hdu::new(
        &[oiii.shape()[1], oiii.shape()[0]],
        oiii.as_slice().unwrap().to_vec()
    );
    Fits::create("oiii.fit", oiii_hdu).expect("Failed to write OIII FITS file");
}

fn optimized_genome(
    camera: &Camera,
    red: &Array2<f64>,
    green: &Array2<f64>,
    blue: &Array2<f64>,
) -> Genome {
    let context = Context::new_with_options(Options::new()).unwrap();

    let image = context
        .new_image(
            &ArrayF64D1::new(
                &context,
                [red.len() as i64],
                red.flatten().as_slice().unwrap(),
            ).unwrap(),
            &ArrayF64D1::new(
                &context,
                [green.len() as i64],
                green.flatten().as_slice().unwrap(),
            ).unwrap(),
            &ArrayF64D1::new(
                &context,
                [blue.len() as i64],
                blue.flatten().as_slice().unwrap(),
            ).unwrap(),
        )
        .unwrap();

    let qe_red = camera.qe_red.as_gpu_qe(&context);
    let qe_green = camera.qe_green.as_gpu_qe(&context);
    let qe_blue = camera.qe_blue.as_gpu_qe(&context);

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
                LAMBDA,
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
                ha_r: parent.ha_r + rng.sample(NormalDistribution::new(0.0, mutation_rate)),
                ha_g: parent.ha_g + rng.sample(NormalDistribution::new(0.0, mutation_rate)),
                ha_b: parent.ha_b + rng.sample(NormalDistribution::new(0.0, mutation_rate)),
                oiii_r: parent.oiii_r + rng.sample(NormalDistribution::new(0.0, mutation_rate)),
                oiii_g: parent.oiii_g + rng.sample(NormalDistribution::new(0.0, mutation_rate)),
                oiii_b: parent.oiii_b + rng.sample(NormalDistribution::new(0.0, mutation_rate)),
            };
            new_population.push(child);
        }

        population = new_population;
        let (best_genome, best_fitness) = best_genome_and_fitness(&population, &fitnesses);
        println!(
            r"
Generation {}:
    Best fitness = {}
    Noise = {}
        ",
            gen,
            best_fitness,
            context
                .noise_fitness(&best_genome.as_gpu_genome(&context), &image)
                .unwrap()
        );
    }

    let (best_genome, best_fitness) = best_genome_and_fitness(&population, &fitnesses);
    println!("Best genome found: {:?}", best_genome);
    println!("Fitness: {}", best_fitness);
    println!(
        "Noise fitness: {}",
        context
            .noise_fitness(&best_genome.as_gpu_genome(&context), &image)
            .unwrap()
    );
    best_genome
}

fn best_genome_and_fitness(population: &Vec<Genome>, fitnesses: &Vec<f64>) -> (Genome, f64) {
    let (best_idx, _) = fitnesses
        .iter()
        .enumerate()
        .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (population[best_idx], fitnesses[best_idx])
}
