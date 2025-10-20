use rand::Rng;
use crate::bindings::{Context, GenomeList};

#[derive(Debug, Copy, Clone)]
pub struct Genome {
    pub(crate) ha_a: f64,
    pub(crate) ha_b: f64,
    pub(crate) ha_c: f64,
    pub(crate) oiii_a: f64,
    pub(crate) oiii_b: f64,
    pub(crate) oiii_c: f64
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            ha_a: rng.random_range(-1.0..1.0),
            ha_b: rng.random_range(-1.0..1.0),
            ha_c: rng.random_range(-1.0..1.0),
            oiii_a: rng.random_range(-1.0..1.0),
            oiii_b: rng.random_range(-1.0..1.0),
            oiii_c: rng.random_range(-1.0..1.0)
        }
    }

    pub fn list_as_gpu_list<'a>(context: &'a Context, list: &Vec<Genome>) -> GenomeList<'a> {
        let mut genomes = context.new_genomes().unwrap();
        for genome in list {
            genomes = context.add_genome(
                &genomes,
                genome.ha_a,
                genome.ha_b,
                genome.ha_c,
                genome.oiii_a,
                genome.oiii_b,
                genome.oiii_c
            ).unwrap();
        }
        genomes
    }
}