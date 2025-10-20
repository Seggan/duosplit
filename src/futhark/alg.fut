type Genome = {
    ha_a: f64,
    ha_b: f64,
    ha_c: f64,
    oiii_a: f64,
    oiii_b: f64,
    oiii_c: f64
}

type GenomeList [n] = [n]Genome

entry new_genomes : GenomeList[] = []

entry add_genome (genomes: GenomeList[]) (ha_a: f64) (ha_b: f64) (ha_c: f64) (oiii_a: f64) (oiii_b: f64) (oiii_c: f64) : GenomeList[] =
    let genome = {ha_a = ha_a, ha_b = ha_b, ha_c = ha_c, oiii_a = oiii_a, oiii_b = oiii_b, oiii_c = oiii_c}
    in genomes ++ [genome]

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

def pixelNoise (a: f64) (b: f64) (c: f64) (pixel: Pixel) : f64 =
    a * a * pixel.r + b * b * pixel.g + c * c * pixel.b

def noise [l] (a: f64) (b: f64) (c: f64) (image: Image[l]) : f64 =
    reduce (+) 0.0 (map (\pixel -> pixelNoise a b c pixel) image) / f64.i64 (length image)

def constraintFitness (genome: Genome) (qeR: QE) (qeG: QE) (qeB: QE) : f64 =
    let c1 = (qeR.ha * genome.ha_a + qeG.ha * genome.ha_b + qeB.ha * genome.ha_c - 1.0)
    let c2 = (qeR.oiii * genome.ha_a + qeG.oiii * genome.ha_b + qeB.oiii * genome.ha_c)
    let c3 = (qeR.ha * genome.oiii_a + qeG.ha * genome.oiii_b + qeB.ha * genome.oiii_c)
    let c4 = (qeR.oiii * genome.oiii_a + qeG.oiii * genome.oiii_b + qeB.oiii * genome.oiii_c - 1.0)
    in c1 * c1 + c2 * c2 + c3 * c3 + c4 * c4

def fitnessOne [l] (genome: Genome) (image: Image[l]) (lambda: f64) (qeR: QE) (qeG: QE) (qeB: QE) : f64 =
    let ha = noise genome.ha_a genome.ha_b genome.ha_c image
    let oiii = noise genome.oiii_a genome.oiii_b genome.oiii_c image
    let con = constraintFitness genome qeR qeG qeB
    in ha + oiii + con * lambda


entry fitness [n][l] (genomes: GenomeList[n]) (image: Image[l]) (lambda: f64) (qeR: QE) (qeG: QE) (qeB: QE) : [n]f64 =
    map (\genome -> fitnessOne genome image lambda qeR qeG qeB) genomes