use fitrs::{Fits, FitsData, Hdu, HeaderValue};
use ndarray::{Array2, Array3};
use std::path::{Path, PathBuf};

pub fn read_fits(path: &impl AsRef<Path>) -> Result<Array3<f32>, String> {
    let image = Fits::open(path).map_err(|e| format!("Failed to open FITS file: {}", e))?;
    let hdu = image.get(0).ok_or("No HDU found in FITS file")?;
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
        .mapv(|v| (v * scale + offset) as f32);
    Ok(channels)
}

pub fn write_fits(path: &PathBuf, data: &Array2<f32>) -> Result<(), String> {
    let hdu = Hdu::new(
        &[data.shape()[1], data.shape()[0]],
        data.as_slice().unwrap().to_vec(),
    );
    Fits::create(path, hdu)
        .map(|_| ())
        .map_err(|e| format!("Failed to write to {}: {}", path.to_str().unwrap(), e))
}