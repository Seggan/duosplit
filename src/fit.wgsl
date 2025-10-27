// R = aH + bO
// G = cH + dO
// B = eH + fO

// H = iR + jG + kB
// O = xR + yG + zB

// H = H*(ai+cj+ek) + O*(bi+dj+fk)
// O = H*(ax+cy+ez) + O*(bx+dy+fz)

// Hnoise^2 = i^2R + j^2G + k^2B
// Onoise^2 = x^2R + y^2G + z^2B

// ai + cj + ek = 1
// bi + dj + fk = 0
// ax + cy + ez = 0
// bx + dy + fz = 1

// j = (d + b c i - a d i)/(d e - c f)
// k = (-f - b e i + a f i)/(d e - c f)

struct QE {
    ha: f32,
    oiii: f32
};

struct Genome {
    i: f32,
    x: f32
};

@group(0) @binding(0) var<storage, read> genomes: array<Genome>;
@group(0) @binding(1) var<storage, read_write> fitness: array<f32>;
@group(0) @binding(2) var<storage, read> image: array<vec3f>;
@group(0) @binding(3) var<uniform> qeR: QE;
@group(0) @binding(4) var<uniform> qeG: QE;
@group(0) @binding(5) var<uniform> qeB: QE;
@group(0) @binding(6) var<uniform> total_chunks: u32;

fn j_k_from_i(i: f32, a: f32, c: f32, e: f32, b: f32, d: f32, f: f32) -> vec2f {
    let denom = d * e - c * f;
    let j = (d + b * c * i - a * d * i) / denom;
    let k = (-f - b * e * i + a * f * i) / denom;
    return vec2f(j, k);
}

fn pixel_noise(a: f32, b: f32, c: f32, pixel: vec3f) -> f32 {
    return a * a * pixel.r + b * b * pixel.g + c * c * pixel.b;
}

@compute @workgroup_size(4, 64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let genome_idx = gid.x;
    let chunk = gid.y;
    if (genome_idx >= arrayLength(&genomes) || chunk >= total_chunks) {
        return;
    }
    let genome = genomes[genome_idx];

    let chunk_size = (arrayLength(&image) + total_chunks - 1u) / total_chunks;
    var fitness_value: f32 = 0.0;
    for (var idx: u32 = chunk * chunk_size; idx < (chunk + 1u) * chunk_size && idx < arrayLength(&image); idx = idx + 1u) {
        let pixel = image[idx];
        let jk = j_k_from_i(genome.i, qeR.ha, qeG.ha, qeB.ha, qeR.oiii, qeG.oiii, qeB.oiii);
        let yz = j_k_from_i(genome.x, qeR.oiii, qeG.oiii, qeB.oiii, qeR.ha, qeG.ha, qeB.ha);

        let h = pixel_noise(genome.i, jk.x, jk.y, pixel);
        let o = pixel_noise(genome.x, yz.x, yz.y, pixel);

        fitness_value += h * h + o * o;
    }
    fitness[genome_idx * total_chunks + chunk] = fitness_value;
}