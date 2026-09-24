use crate::cli::Cli;
use crate::fits::write_fits;
use crate::genetics::TileProcessor;
use crate::gpu::GpuDevice;
use anyhow::Result;
use clap::Parser;
use ndarray::Array2;
use std::process::exit;

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

    println!("Starting genetic algorithm optimization...");
    let mut h_alpha = Array2::default((image.shape()[0], image.shape()[1]));
    let mut oiii = Array2::default((image.shape()[0], image.shape()[1]));

    let tiles = TileProcessor::gen_tiles(&cli, &device, &image)?;
    for (i, mut tile) in tiles.into_iter().enumerate() {
        println!("Tile {}", i);
        tile.run_genetic_algorithm(cli.timings)?;
        tile.output(&mut h_alpha, &mut oiii);
    }

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
