type Genome = {
    i: f64,
    x: f64
}

type Genomes [n] = [n]Genome

entry new_genomes : Genomes[] = []

entry add_genome [n] (genomes: Genomes[n]) (genome: Genome) : Genomes[n + 1] =
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

-- j = (d + b c i - a d i)/(d e - c f)
-- k = (-f - b e i + a f i)/(d e - c f)

entry j_k_from_i (i: f64) (a: f64) (c: f64) (e: f64) (b: f64) (d: f64) (f: f64) : (f64, f64) =
    let denom = d * e - c * f
    let j = (d + b * c * i - a * d * i) / denom
    let k = (-f - b * e * i + a * f * i) / denom
    in (j, k)

def pixel_noise (a: f64) (b: f64) (c: f64) (pixel: Pixel) : f64 =
    a * a * pixel.r + b * b * pixel.g + c * c * pixel.b

def noise [l] (i: f64) (j: f64) (k: f64) (image: Image[l]) : f64 =
    reduce (+) 0.0 (map (\pixel -> pixel_noise i j k pixel) image) / f64.i64 (length image)

def fitness_one [l] (genome: Genome) (image: Image[l]) (qeR: QE) (qeG: QE) (qeB: QE) : f64 =
    let {i, x} = genome
    let (j, k) = j_k_from_i i qeR.ha qeG.ha qeB.ha qeR.oiii qeG.oiii qeB.oiii
    let (y, z) = j_k_from_i x qeR.oiii qeG.oiii qeB.oiii qeR.ha qeG.ha qeB.ha
    let h_noise = noise i j k image
    let o_noise = noise x y z image
    in h_noise + o_noise

entry fitness [n][l] (genomes: Genomes[n]) (image: Image[l]) (qeR: QE) (qeG: QE) (qeB: QE) : [n]f64 =
    map (\genome -> fitness_one genome image qeR qeG qeB) genomes