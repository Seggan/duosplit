# duosplit

A tool for accurately splitting hydrogen-alpha and oxygen-III images from dual narrowband astrophotography captures.
Based on the math provided (and explained) to me by Corvusmellori on the Astrophotography & Co. Discord.

## Downloads
Windows, MacOS, and Linux binaries are available on the [releases page](https://github.com/Seggan/duosplit/releases).

## Usage
```
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
  -h, --help
          Print help
  -V, --version
          Print version
```