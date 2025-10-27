use std::path::PathBuf;
use clap::Parser;

#[derive(Parser)]
#[command(version, about)]
pub struct Cli {
    #[arg(help = "Path to input FITS file")]
    pub input: PathBuf,

    #[arg(short, long, default_value = ".", help = "Path to output directory")]
    pub output: PathBuf,

    #[arg(long = "qrh", help = "The quantum efficiency of the red channel at the hydrogen-alpha wavelength (656.3 nm)")]
    pub red_ha_qe: f32,

    #[arg(long = "qgh", help = "The quantum efficiency of the green channel at the hydrogen-alpha wavelength (656.3 nm)")]
    pub green_ha_qe: f32,

    #[arg(long = "qbh", help = "The quantum efficiency of the blue channel at the hydrogen-alpha wavelength (656.3 nm)")]
    pub blue_ha_qe: f32,

    #[arg(long = "qro", help = "The quantum efficiency of the red channel at the OIII wavelength (500.7 nm)")]
    pub red_oiii_qe: f32,

    #[arg(long = "qgo", help = "The quantum efficiency of the green channel at the OIII wavelength (500.7 nm)")]
    pub green_oiii_qe: f32,

    #[arg(long = "qbo", help = "The quantum efficiency of the blue channel at the OIII wavelength (500.7 nm)")]
    pub blue_oiii_qe: f32,

    #[arg(short, long, default_value_t = 100, help = "Population size for the genetic algorithm")]
    pub population_size: usize,

    #[arg(short, long, default_value_t = 250, help = "Number of generations for the genetic algorithm")]
    pub generations: u32,

    #[arg(short, long, default_value_t = 5, help = "Number of elite individuals to carry over each generation")]
    pub elitism: usize,

    #[arg(short = 's', long, default_value_t = 0.5, help = "Initial standard deviation for mutation")]
    pub initial_std: f32,
    
    #[arg(short, long, default_value_t = 0.1, help = "Decay rate for mutation standard deviation")]
    pub decay_rate: f32,

    #[arg(short, long, action)]
    pub timings: bool
}