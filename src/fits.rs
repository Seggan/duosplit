use crate::gpu::{BufferBindingSpec, GpuDevice};
use anyhow::{anyhow, Context, Result};
use fitrs::{Fits, FitsData, Hdu, HeaderValue};
use ndarray::{Array2, Array3};
use std::path::{Path, PathBuf};
use wgpu::BufferBindingType;

pub fn read_fits(path: &impl AsRef<Path>) -> Result<Array3<f32>> {
    let image = Fits::open(path)?;
    let hdu = image.get(0).ok_or(anyhow!("No HDU found in FITS file"))?;
    let scale = hdu
        .value("BSCALE")
        .map(|v| match v {
            HeaderValue::IntegerNumber(i) => *i as f64,
            HeaderValue::RealFloatingNumber(f) => *f,
            _ => panic!("Unexpected BSCALE type"),
        })
        .unwrap_or(1.0);
    let offset = hdu
        .value("BZERO")
        .map(|v| match v {
            HeaderValue::IntegerNumber(i) => *i as f64,
            HeaderValue::RealFloatingNumber(f) => *f,
            _ => panic!("Unexpected BZERO type"),
        })
        .unwrap_or(0.0);
    let (shape, data) = match hdu.read_data() {
        FitsData::Characters(arr) => (
            arr.shape,
            arr.data.into_iter().map(|v| v as u64 as f64).collect(),
        ),
        FitsData::IntegersI32(arr) => (
            arr.shape,
            arr.data
                .into_iter()
                .map(|v| v.unwrap_or(0) as f64)
                .collect(),
        ),
        FitsData::IntegersU32(arr) => (
            arr.shape,
            arr.data
                .into_iter()
                .map(|v| v.unwrap_or(0) as f64)
                .collect(),
        ),
        FitsData::FloatingPoint32(arr) => {
            (arr.shape, arr.data.into_iter().map(|v| v as f64).collect())
        }
        FitsData::FloatingPoint64(arr) => {
            eprintln!(
                "Warning: Converting FITS data from 64 bit to 32 bit; this may lose precision."
            );
            (arr.shape, arr.data)
        }
    };

    let channels = Array3::from_shape_vec((shape[2], shape[1], shape[0]), data)
        .expect("Failed to reshape FITS data into 3D array")
        .reversed_axes()
        .mapv(|v| (v * scale + offset) as f32)
        .as_standard_layout()
        .into_owned();
    Ok(channels)
}

pub fn write_fits(path: &PathBuf, data: Array2<f32>) -> Result<()> {
    let hdu = Hdu::new(
        &[data.shape()[0], data.shape()[1]],
        data.reversed_axes().as_standard_layout().as_slice().unwrap().to_vec(),
    );
    Fits::create(path, hdu)
        .map(|_| ())
        .with_context(|| format!("Failed to write to {}", path.to_str().unwrap()))
}

// pub fn mmt(image: &Array3<f32>) -> Array3<f32> {
//
// }

pub fn lum(device: &GpuDevice, image: &Array3<f32>) -> Result<Array2<f32>> {
    let flat = image.flatten();
    let flat = flat.as_standard_layout();

    let width = image.shape()[0] as u32;
    let height = image.shape()[1] as u32;

    let mut context = device.create_program(
        include_str!("lum.wgsl"),
        vec![
            BufferBindingSpec {
                name: "image_in".into(),
                binding_type: BufferBindingType::Storage { read_only: true },
                min_size: (flat.len() * size_of::<f32>()) as u32,
                allow_readout: false,
            },
            BufferBindingSpec {
                name: "image_out".into(),
                binding_type: BufferBindingType::Storage { read_only: false },
                min_size: width * height * size_of::<f32>() as u32,
                allow_readout: true,
            },
            BufferBindingSpec {
                name: "dims".into(),
                binding_type: BufferBindingType::Uniform,
                min_size: 0,
                allow_readout: false,
            },
        ],
    )?;

    context
        .get_buffer_binding("image_in")
        .unwrap()
        .set_data(bytemuck::cast_slice(flat.as_slice().unwrap()));

    context
        .get_buffer_binding("image_out")
        .unwrap()
        .set_data(bytemuck::cast_slice(&vec![f32::NAN; flat.len() / 3]));

    context
        .get_buffer_binding("dims")
        .unwrap()
        .set_data(bytemuck::cast_slice(&vec![
            width,
            height,
        ]));

    context.execute((
        width,
        height,
        1,
    ))?;

    let data = context
        .get_buffer_binding("image_out")
        .unwrap()
        .read_data()
        .unwrap()
        .chunks(size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    let actual_data = Array2::from_shape_vec((width as usize, height as usize), data)?;
    Ok(actual_data)
}
