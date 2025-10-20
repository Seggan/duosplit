type Genome = {
    haA: f64,
    haB: f64,
    haC: f64,
    oiiiA: f64,
    oiiiB: f64,
    oiiiC: f64
}

type Pixel = {
    r: f64,
    g: f64,
    b: f64
}

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

def noise [l] (a: f64) (b: f64) (c: f64) (image: [l]Pixel) : f64 =
    reduce (+) 0.0 (map (\pixel -> pixelNoise a b c pixel) image)

def constraintFitness (genome: Genome) (qeR: QE) (qeG: QE) (qeB: QE) : f64 =
    let c1 = (qeR.ha * genome.haA + qeG.ha * genome.haB + qeB.ha * genome.haC - 1.0)
    let c2 = (qeR.oiii * genome.haA + qeG.oiii * genome.haB + qeB.oiii * genome.haC)
    let c3 = (qeR.ha * genome.oiiiA + qeG.ha * genome.oiiiB + qeB.ha * genome.oiiiC)
    let c4 = (qeR.oiii * genome.oiiiA + qeG.oiii * genome.oiiiB + qeB.oiii * genome.oiiiC - 1.0)
    in c1 * c1 + c2 * c2 + c3 * c3 + c4 * c4

def fitnessOne [l] (genome: Genome) (image: [l]Pixel) (lambda: f64) (qeR: QE) (qeG: QE) (qeB: QE) : f64 =
    let ha = noise genome.haA genome.haB genome.haC image
    let oiii = noise genome.oiiiA genome.oiiiB genome.oiiiC image
    let con = constraintFitness genome qeR qeG qeB
    in ha + oiii + con * lambda

entry fitness [n][l] (genomes: [n]Genome) (image: [l]Pixel) (lambda: f64) (qeR: QE) (qeG: QE) (qeB: QE) : [n]f64 =
    map (\genome -> fitnessOne genome image lambda qeR qeG qeB) genomes