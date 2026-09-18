use crate::genetics::Genome;
use crate::gpu::{BufferBindingSpec, GpuContext, GpuContextCreationError};
use bytemuck::{Pod, Zeroable};
use wgpu::BufferBindingType;

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct QEUniform {
    pub ha: f32,
    pub oiii: f32,
}

pub struct FitnessCalculator {
    context: GpuContext,
    chunks: usize,
    image_len: usize,
}

impl FitnessCalculator {
    pub async fn new(
        image: Vec<[f32; 3]>,
        chunks: usize,
        quantum_efficiencies: (QEUniform, QEUniform, QEUniform),
    ) -> Result<Self, String> {
        let mut context = match GpuContext::new(
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
        )
            .await
        {
            Ok(context) => context,
            Err(e) => return Err(match e {
                GpuContextCreationError::BindingTooLarge(_) => "Image chunk size exceeds maximum buffer size for the GPU adapter. You must increase the chunk amount in order to process the image".to_string(),
                _ => e.to_string()
            }),
        };

        context
            .get_buffer_binding(IMAGE_BINDING)
            .unwrap()
            .set_data(bytemuck::cast_slice(&image));

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

    pub fn get_gpu_name(&self) -> &str {
        self.context.get_gpu_name()
    }

    pub async fn compute_fitness(&mut self, genomes: &[Genome]) -> Result<Vec<f32>, String> {
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
            .execute((workgroup_count_x, workgroup_count_y, 1))
            .await
            .map_err(|err| err.to_string())?;

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
