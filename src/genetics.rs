use rand::Rng;
use crate::bindings;
use crate::bindings::{Context, GenomeList};

#[derive(Debug, Copy, Clone)]
pub struct Genome {
    pub(crate) ha_r: f64,
    pub(crate) ha_g: f64,
    pub(crate) ha_b: f64,
    pub(crate) oiii_r: f64,
    pub(crate) oiii_g: f64,
    pub(crate) oiii_b: f64
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            ha_r: rng.random_range(-1.0..1.0),
            ha_g: rng.random_range(-1.0..1.0),
            ha_b: rng.random_range(-1.0..1.0),
            oiii_r: rng.random_range(-1.0..1.0),
            oiii_g: rng.random_range(-1.0..1.0),
            oiii_b: rng.random_range(-1.0..1.0)
        }
    }

    pub fn list_as_gpu_list<'a>(context: &'a Context, list: &Vec<Genome>) -> GenomeList<'a> {
        let mut genomes = context.new_genomes().unwrap();
        for genome in list {
            genomes = context.add_genome(&genomes, &genome.as_gpu_genome(context)).unwrap();
        }
        genomes
    }

    pub fn as_gpu_genome(self, context: &Context) -> bindings::Genome {
        bindings::Genome::new(
            context,
            self.ha_r,
            self.ha_g,
            self.ha_b,
            self.oiii_r,
            self.oiii_g,
            self.oiii_b
        ).unwrap()
    }
}