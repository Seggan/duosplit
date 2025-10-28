# duosplit

A tool for accurately splitting hydrogen-alpha and oxygen-III images from dual narrowband astrophotography captures.
Based on the math provided (and explained) to me by Raven LaRue.

## Downloads
Windows and Linux binaries are available on the [releases page](https://github.com/Seggan/duosplit/releases).
Due to MacOS being a super annoying platform to build for and I don't have a Mac, MacOS binaries are not provided.
You can build from source on MacOS using the instructions below.

## Siril Script
I also have created a Siril script that automatically downloads the correct runtimes and runs duosplit for you on the Siril image.
The script is also located on the releases page.

## Using Linux, Wayland, and Vulkan
WGPU (the graphical computation backend used by duosplit) has some issues on Linux with Wayland and Vulkan.
If you are using Wayland, you may need to set the environment variable `WGPU_BACKEND` to `gl` to use the OpenGL backend instead of Vulkan.
If you are using the Siril script, it will automatically set this variable for you.

## Building from Source
Building duosplit requires [Rust](https://www.rust-lang.org/) and [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) to be installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/Seggan/duosplit.git
   cd duosplit
   ```
   
2. Build the project:
   ```bash
   cargo build --release
   ```
   
3. The compiled binary will be located in the `target/release` directory.

## Usage
```
A tool for splitting dual-narrowband hydrogen-alpha and oxygen-III images.

Usage: duosplit [OPTIONS] --qrh <RED_HA_QE> --qgh <GREEN_HA_QE> --qbh <BLUE_HA_QE> --qro <RED_OIII_QE> --qgo <GREEN_OIII_QE> --qbo <BLUE_OIII_QE> <INPUT>

Arguments:
  <INPUT>  Path to input FITS file

Options:
  -o, --output <OUTPUT>
          Path to output directory [default: .]
      --qrh <RED_HA_QE>
          The quantum efficiency of the red channel at the hydrogen-alpha wavelength (656.3 nm)
      --qgh <GREEN_HA_QE>
          The quantum efficiency of the green channel at the hydrogen-alpha wavelength (656.3 nm)
      --qbh <BLUE_HA_QE>
          The quantum efficiency of the blue channel at the hydrogen-alpha wavelength (656.3 nm)
      --qro <RED_OIII_QE>
          The quantum efficiency of the red channel at the OIII wavelength (500.7 nm)
      --qgo <GREEN_OIII_QE>
          The quantum efficiency of the green channel at the OIII wavelength (500.7 nm)
      --qbo <BLUE_OIII_QE>
          The quantum efficiency of the blue channel at the OIII wavelength (500.7 nm)
  -p, --population-size <POPULATION_SIZE>
          Population size for the genetic algorithm [default: 100]
  -g, --generations <GENERATIONS>
          Number of generations for the genetic algorithm [default: 250]
  -e, --elitism <ELITISM>
          Number of elite individuals to carry over each generation [default: 5]
  -s, --initial-std <INITIAL_STD>
          Initial standard deviation for mutation [default: 0.5]
  -d, --decay-rate <DECAY_RATE>
          Decay rate for mutation standard deviation [default: 0.1]
  -c, --chunks <CHUNKS>
          Number of chunks to split the image into before processing on the GPU [default: 2048]
  -t, --timings
          Enable timing output
  -h, --help
          Print help
  -V, --version
          Print version
```