use crate::bindings;
use crate::bindings::{Context, Genomes};
use rand::Rng;

#[derive(Debug, Copy, Clone)]
pub struct Genome {
    pub i: f64,
    pub x: f64,
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            i: rng.random_range(-1.0..1.0),
            x: rng.random_range(-1.0..1.0)
        }
    }

    pub fn list_as_gpu_list<'a>(context: &'a Context, list: &Vec<Genome>) -> Genomes<'a> {
        let mut genomes = context.new_genomes().unwrap();
        for genome in list {
            genomes = context.add_genome(&genomes, &genome.as_gpu_genome(context)).unwrap();
        }
        genomes
    }

    pub fn as_gpu_genome(self, context: &Context) -> bindings::Genome {
        bindings::Genome::new(
            context,
            self.i,
            self.x,
        ).unwrap()
    }
}