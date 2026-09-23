@group(0) @binding(0) var<storage, read> image_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> image_out: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    let height = dims.x;
    let width = dims.y;

    if (x >= width || y >= height) {
        return;
    }

    let pixel_idx = y * width + x;
    let input_idx = pixel_idx * 3u;

    let r = image_in[input_idx + 0u];
    let g = image_in[input_idx + 1u];
    let b = image_in[input_idx + 2u];

    image_out[pixel_idx] =
        0.2126 * r +
        0.7152 * g +
        0.0722 * b;
}
