use crate::genetics::Genome;
use crate::gpu::{BufferBindingSpec, GpuDevice, GpuProgram};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use ndarray::Array3;
use wgpu::BufferBindingType;

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct QEUniform {
    pub ha: f32,
    pub oiii: f32,
}

pub struct FitnessCalculator {
    context: GpuProgram,
    chunks: usize,
    image_len: usize,
}

impl FitnessCalculator {
    pub fn new(
        device: &GpuDevice,
        image: &Array3<f32>,
        chunks: usize,
        quantum_efficiencies: (QEUniform, QEUniform, QEUniform),
    ) -> Result<Self> {
        let mut context = device.create_program(
            include_str!("fit.wgsl"),
            vec![
                BufferBindingSpec {
                    name: GENOMES_BINDING.to_string(),
                    binding_type: BufferBindingType::Storage { read_only: true },
                    min_size: 0,
                    allow_readout: false,
                },
                BufferBindingSpec {
                    name: FITNESS_BINDING.to_string(),
                    binding_type: BufferBindingType::Storage { read_only: false },
                    min_size: 0,
                    allow_readout: true,
                },
                BufferBindingSpec {
                    name: IMAGE_BINDING.to_string(),
                    binding_type: BufferBindingType::Storage { read_only: true },
                    min_size: (image.len() * size_of::<[f32; 3]>() / chunks) as u32,
                    allow_readout: false,
                },
                BufferBindingSpec {
                    name: QE_R_BINDING.to_string(),
                    binding_type: BufferBindingType::Uniform,
                    min_size: 0,
                    allow_readout: false,
                },
                BufferBindingSpec {
                    name: QE_G_BINDING.to_string(),
                    binding_type: BufferBindingType::Uniform,
                    min_size: 0,
                    allow_readout: false,
                },
                BufferBindingSpec {
                    name: QE_B_BINDING.to_string(),
                    binding_type: BufferBindingType::Uniform,
                    min_size: 0,
                    allow_readout: false,
                },
                BufferBindingSpec {
                    name: CHUNKS_BINDING.to_string(),
                    binding_type: BufferBindingType::Uniform,
                    min_size: 0,
                    allow_readout: false,
                },
            ],
        )?;

        context
            .get_buffer_binding(IMAGE_BINDING)
            .unwrap()
            .set_data(bytemuck::cast_slice(&image.as_slice().unwrap()));

        context
            .get_buffer_binding(QE_R_BINDING)
            .unwrap()
            .set_data(bytemuck::bytes_of(&quantum_efficiencies.0));

        context
            .get_buffer_binding(QE_G_BINDING)
            .unwrap()
            .set_data(bytemuck::bytes_of(&quantum_efficiencies.1));

        context
            .get_buffer_binding(QE_B_BINDING)
            .unwrap()
            .set_data(bytemuck::bytes_of(&quantum_efficiencies.2));

        context
            .get_buffer_binding(CHUNKS_BINDING)
            .unwrap()
            .set_data(bytemuck::bytes_of(&chunks));

        Ok(Self {
            context,
            chunks,
            image_len: image.len(),
        })
    }

    pub fn compute_fitness(&mut self, genomes: &[Genome]) -> Result<Vec<f32>> {
        self.context
            .get_buffer_binding(GENOMES_BINDING)
            .unwrap()
            .set_data(bytemuck::cast_slice(&genomes));

        self.context
            .get_buffer_binding(FITNESS_BINDING)
            .unwrap()
            .set_data(bytemuck::cast_slice(&vec![
                0.0f32;
                genomes.len() * self.chunks
            ]));

        let workgroup_count_x = ((genomes.len() as f32) / 4.0).ceil() as u32;
        let workgroup_count_y = ((self.chunks as f32) / 64.0).ceil() as u32;
        self.context
            .execute((workgroup_count_x, workgroup_count_y, 1))?;

        let fitness_binding = self.context.get_buffer_binding(FITNESS_BINDING).unwrap();
        let data = bytemuck::cast_slice(&fitness_binding.read_data().unwrap()).to_vec();
        Ok(data
            .chunks(self.chunks)
            .map(|chunk| chunk.iter().sum::<f32>())
            .map(|fit| fit / (self.image_len as f32))
            .collect())
    }
}

const GENOMES_BINDING: &'static str = "genomes";
const FITNESS_BINDING: &'static str = "fitnesses";
const IMAGE_BINDING: &'static str = "image";
const QE_R_BINDING: &'static str = "qe-r";
const QE_G_BINDING: &'static str = "qe-g";
const QE_B_BINDING: &'static str = "qe-b";
const CHUNKS_BINDING: &'static str = "chunks";
