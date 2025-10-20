type Genome = {
    ha_r: f64,
    ha_g: f64,
    ha_b: f64,
    oiii_r: f64,
    oiii_g: f64,
    oiii_b: f64
}

entry new_genome (ha_r: f64) (ha_g: f64) (ha_b: f64)
                     (oiii_r: f64) (oiii_g: f64) (oiii_b: f64) : Genome =
    { ha_r = ha_r,
      ha_g = ha_g,
      ha_b = ha_b,
      oiii_r = oiii_r,
      oiii_g = oiii_g,
      oiii_b = oiii_b }

type GenomeList [n] = [n]Genome

entry new_genomes : GenomeList[] = []

entry add_genome (genomes: GenomeList[]) (genome: Genome) : GenomeList[] =
    genomes ++ [genome]

type Pixel = {
    r: f64,
    g: f64,
    b: f64
}

type Image [l] = [l]Pixel

entry new_image [l] (rs: [l]f64) (gs: [l]f64) (bs: [l]f64) : Image[l] =
    map3 (\r g b -> {r = r, g = g, b = b}) rs gs bs

type QE = {
    ha: f64,
    oiii: f64
}

-- R = aH + bO
-- G = cH + dO
-- B = eH + fO

-- H = iR + jG + kB
-- O = xR + yG + zB

-- H = H*(ai+cj+ek) + O*(bi+dj+fk)
-- O = H*(ax+cy+ez) + O*(bx+dy+fz)

-- Hnoise^2 = i^2R + j^2G + k^2B
-- Onoise^2 = x^2R + y^2G + z^2B

-- ai + cj + ek = 1
-- bi + dj + fk = 0
-- ax + cy + ez = 0
-- bx + dy + fz = 1

def pixel_noise (a: f64) (b: f64) (c: f64) (pixel: Pixel) : f64 =
    a * a * pixel.r + b * b * pixel.g + c * c * pixel.b

def noise [l] (a: f64) (b: f64) (c: f64) (image: Image[l]) : f64 =
    reduce (+) 0.0 (map (\pixel -> pixel_noise a b c pixel) image) / f64.i64 (length image)

def constraint_fitness (genome: Genome) (qeR: QE) (qeG: QE) (qeB: QE) : f64 =
    let c1 = qeR.ha * genome.ha_r + qeG.ha * genome.ha_g + qeB.ha * genome.ha_b - 1.0
    let c2 = qeR.oiii * genome.ha_r + qeG.oiii * genome.ha_g + qeB.oiii * genome.ha_b
    let c3 = qeR.ha * genome.oiii_r + qeG.ha * genome.oiii_g + qeB.ha * genome.oiii_b
    let c4 = qeR.oiii * genome.oiii_r + qeG.oiii * genome.oiii_g + qeB.oiii * genome.oiii_b - 1.0
    in c1 * c1 + c2 * c2 + c3 * c3 + c4 * c4

entry noise_fitness (genome: Genome) (image: Image[]) : f64 =
    let ha = noise genome.ha_r genome.ha_g genome.ha_b image
    let oiii = noise genome.oiii_r genome.oiii_g genome.oiii_b image
    in ha + oiii

def fitness_one [l] (genome: Genome) (image: Image[l]) (lambda: f64) (qeR: QE) (qeG: QE) (qeB: QE) : f64 =
    let noise_fit = noise_fitness genome image
    let constraint_fit = constraint_fitness genome qeR qeG qeB
    in noise_fit + lambda * constraint_fit

entry fitness [n][l] (genomes: GenomeList[n]) (image: Image[l]) (lambda: f64) (qeR: QE) (qeG: QE) (qeB: QE) : [n]f64 =
    map (\genome -> fitness_one genome image lambda qeR qeG qeB) genomes