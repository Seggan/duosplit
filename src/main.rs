use crate::bindings::{ArrayF64D1, Context, Options};
use crate::config::Camera;
use crate::genetics::Genome;
use crate::normal_distr::NormalDistribution;
use fitrs::{Fits, FitsData, Hdu};
use ndarray::{s, Array2, Array3};
use rand::{rng, Rng};
use std::fs::File;
use std::path::Path;

mod bindings;
mod config;
mod genetics;
mod normal_distr;

const POP_SIZE: usize = 100;
const GENS: u32 = 250;
const INITIAL_STD: f64 = 0.5;
const DECAY_RATE: f64 = 0.1;
const ELITISM: usize = 5;

fn main() {
    let config = Path::new("config.json");
    let cameras = serde_json::from_reader::<_, Vec<Camera>>(
        File::open(config).expect("Failed to open config file"),
    )
    .expect("Failed to parse config file");
    let camera = &cameras[0];
    println!("Using camera: {:?}", camera);

    let image = Fits::open("image_real.fit").expect("Failed to open FITS file");
    let (shape, data) = match image.get(0).expect("No HDU found").read_data() {
        FitsData::Characters(_) => panic!("Did not expect character data"),
        FitsData::IntegersI32(arr) => (
            arr.shape,
            arr.data.into_iter().map(|v| v.unwrap_or(0) as f64).collect(),
        ),
        FitsData::IntegersU32(arr) => (
            arr.shape,
            arr.data.into_iter().map(|v| v.unwrap_or(0) as f64).collect(),
        ),
        FitsData::FloatingPoint32(arr) => (
            arr.shape,
            arr.data.into_iter().map(|v| v as f64).collect(),
        ),
        FitsData::FloatingPoint64(arr) => (arr.shape, arr.data),
    };
    let channels = Array3::from_shape_vec((shape[2], shape[1], shape[0]), data).unwrap();

    let red_channel = channels.slice(s![0, .., ..]).into_owned();
    let green_channel = channels.slice(s![1, .., ..]).into_owned();
    let blue_channel = channels.slice(s![2, .., ..]).into_owned();

    let context = Context::new_with_options(Options::new()).unwrap();
    let best_genome = optimized_genome(
        &context,
        camera,
        &red_channel,
        &green_channel,
        &blue_channel,
    );

    let ha_r = best_genome.i;
    let (ha_g, ha_b) = context.j_k_from_i(
        ha_r,
        camera.qe_red.ha,
        camera.qe_green.ha,
        camera.qe_blue.ha,
        camera.qe_red.oiii,
        camera.qe_green.oiii,
        camera.qe_blue.oiii,
    ).unwrap();
    let h_alpha = ha_r * &red_channel
        + ha_g * &green_channel
        + ha_b * &blue_channel;

    let oiii_r = best_genome.x;
    let (oiii_g, oiii_b) = context.j_k_from_i(
        oiii_r,
        camera.qe_red.oiii,
        camera.qe_green.oiii,
        camera.qe_blue.oiii,
        camera.qe_red.ha,
        camera.qe_green.ha,
        camera.qe_blue.ha,
    ).unwrap();
    let oiii = oiii_r * &red_channel
        + oiii_g * &green_channel
        + oiii_b * &blue_channel;

    println!("Best genome results:");
    println!("H-alpha coefficients: r = {}, g = {}, b = {}", ha_r, ha_g, ha_b);
    println!("OIII coefficients: r = {}, g = {}, b = {}", oiii_r, oiii_g, oiii_b);

    let ha_hdu = Hdu::new(
        &[h_alpha.shape()[1], h_alpha.shape()[0]],
        h_alpha.as_slice().unwrap().to_vec(),
    );
    Fits::create("h_alpha.fit", ha_hdu).expect("Failed to write H-alpha FITS file");

    let oiii_hdu = Hdu::new(
        &[oiii.shape()[1], oiii.shape()[0]],
        oiii.as_slice().unwrap().to_vec(),
    );
    Fits::create("oiii.fit", oiii_hdu).expect("Failed to write OIII FITS file");
}

fn optimized_genome(
    context: &Context,
    camera: &Camera,
    red: &Array2<f64>,
    green: &Array2<f64>,
    blue: &Array2<f64>,
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
        println!(
            r"
Generation {}:
    Noise = {}
        ",
            gen, best_fitness
        );
    }

    let (best_genome, best_fitness) = best_genome_and_fitness(&population, &fitnesses);
    println!("Best genome found: {:?}", best_genome);
    println!("Noise: {}", best_fitness);
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
