use crate::bindings::{ArrayF64D1, Options};
use crate::config::Camera;
use crate::genetics::Genome;
use crate::normal_disr::NormalDistribution;
use fitrs::{Fits, FitsData};
use ndarray::{s, Array3};
use rand::{rng, Rng};
use std::fs::File;
use std::path::Path;

mod bindings;
mod config;
mod genetics;
mod normal_disr;

const POP_SIZE: usize = 100;
const GENS: u32 = 500;
const INITIAL_STD: f64 = 0.05;
const DECAY_RATE: f64 = 0.005;
const ELITISM: usize = 5;
const LAMBDA: f64 = 1e6;

fn main() {
    let context = Box::leak(Box::new(
        bindings::Context::new_with_options(Options::new()).unwrap(),
    ));

    let config = Path::new("config.json");
    let cameras = serde_json::from_reader::<_, Vec<Camera>>(
        File::open(config).expect("Failed to open config file"),
    )
    .expect("Failed to parse config file");
    let camera = &cameras[0];
    println!("Using camera: {:?}", camera);

    let qe_red = camera.qe_red.as_gpu_qe(context);
    let qe_green = camera.qe_green.as_gpu_qe(context);
    let qe_blue = camera.qe_blue.as_gpu_qe(context);

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
    let channels = Array3::from_shape_vec(
        (shape[2], shape[1], shape[0]),
        data
    ).unwrap();

    let red_channel = channels.slice(s![0, .., ..]).into_owned();
    let green_channel = channels.slice(s![1, .., ..]).into_owned();
    let blue_channel = channels.slice(s![2, .., ..]).into_owned();

    let image = context.new_image(
        &ArrayF64D1::new(context, [red_channel.len() as i64], red_channel.flatten().as_slice().unwrap()).unwrap(),
        &ArrayF64D1::new(context, [green_channel.len() as i64], green_channel.flatten().as_slice().unwrap()).unwrap(),
        &ArrayF64D1::new(context, [blue_channel.len() as i64], blue_channel.flatten().as_slice().unwrap()).unwrap()
    ).unwrap();

    let mut rng = rng();
    let mut population = Vec::with_capacity(POP_SIZE);
    for _ in 0..POP_SIZE {
        population.push(Genome::random(&mut rng));
    }

    let mut fitnesses = vec![0.0; POP_SIZE];
    for gen in 0..GENS {
        fitnesses = context.fitness(
            &Genome::list_as_gpu_list(context, &population),
            &image,
            LAMBDA,
            &qe_red,
            &qe_green,
            &qe_blue
        ).unwrap().get().unwrap();

        let elite_indices = {
            let mut indices = (0..POP_SIZE).collect::<Vec<usize>>();
            indices.sort_by(|&i, &j| fitnesses[i].partial_cmp(&fitnesses[j]).unwrap());
            indices[..ELITISM].to_vec()
        };
        let elites = elite_indices.iter().map(|&i| population[i]).collect::<Vec<Genome>>();

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
                oiii_b: parent.oiii_b + rng.sample(NormalDistribution::new(0.0, mutation_rate))
            };
            new_population.push(child);
        }

        population = new_population;
        let (best_genome, best_fitness) = best_genome_and_fitness(&population, &fitnesses);
        println!(r"
Generation {}:
    Best fitness = {}
    Noise = {}
        ", gen, best_fitness, context.noise_fitness(&best_genome.as_gpu_genome(context), &image).unwrap());
    }

    let (best_genome, best_fitness) = best_genome_and_fitness(&population, &fitnesses);
    println!("Best genome found: {:?}", best_genome);
    println!("Fitness: {}", best_fitness);
}

fn best_genome_and_fitness(population: &Vec<Genome>, fitnesses: &Vec<f64>) -> (Genome, f64) {
    let (best_idx, _) = fitnesses.iter().enumerate().min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap()).unwrap();
    (population[best_idx], fitnesses[best_idx])
}