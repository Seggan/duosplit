use crate::genetics::Genome;
use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::PollType;
use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Instance, InstanceDescriptor, MapMode, PipelineLayoutDescriptor,
    Queue, RequestAdapterOptions, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct QEUniform {
    pub ha: f32,
    pub oiii: f32,
}

pub struct GpuContext {
    device: Device,
    queue: Queue,
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
    image_buffer: Buffer,
    chunks: usize,
    image_len: usize,
    quantum_efficiencies: (Buffer, Buffer, Buffer),
}

impl GpuContext {
    pub async fn new(
        image: Vec<[f32; 3]>,
        chunks: usize,
        quantum_efficiencies: (QEUniform, QEUniform, QEUniform),
    ) -> Self {
        let instance = Instance::new(&InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .await
            .unwrap();
        let alg_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("fit.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Genomes
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Fitness
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Image
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // QE uniforms (R, G, B)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Slice and slices
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &alg_shader,
            entry_point: "main".into(),
            compilation_options: Default::default(),
            cache: None,
        });

        let image_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Image Buffer"),
            contents: bytemuck::cast_slice(&image),
            usage: BufferUsages::STORAGE,
        });

        let qe_red_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("QE Red Buffer"),
            contents: bytemuck::bytes_of(&quantum_efficiencies.0),
            usage: BufferUsages::UNIFORM,
        });

        let qe_green_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("QE Green Buffer"),
            contents: bytemuck::bytes_of(&quantum_efficiencies.1),
            usage: BufferUsages::UNIFORM,
        });

        let qe_blue_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("QE Blue Buffer"),
            contents: bytemuck::bytes_of(&quantum_efficiencies.2),
            usage: BufferUsages::UNIFORM,
        });

        Self {
            device,
            queue,
            layout,
            pipeline,
            image_buffer,
            chunks,
            image_len: image.len(),
            quantum_efficiencies: (qe_red_buffer, qe_green_buffer, qe_blue_buffer),
        }
    }

    pub async fn compute_fitness(&self, genomes: &[Genome]) -> Vec<f32> {
        let genome_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Genome Buffer"),
            contents: bytemuck::cast_slice(genomes),
            usage: BufferUsages::STORAGE,
        });

        let fitness = vec![0.0f32; genomes.len() * self.chunks];
        let fitness_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Fitness Buffer"),
            contents: bytemuck::cast_slice(&fitness),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let fitness_staging_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Fitness Staging Buffer"),
            contents: bytemuck::cast_slice(&fitness),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });

        let chunks_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Chunks Buffer"),
            contents: bytemuck::bytes_of(&(self.chunks as u32)),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: genome_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: fitness_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.image_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.quantum_efficiencies.0.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.quantum_efficiencies.1.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: self.quantum_efficiencies.2.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: chunks_buffer.as_entire_binding(),
                },
            ],
            label: None,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count_x = ((genomes.len() as f32) / 4.0).ceil() as u32;
            let workgroup_count_y = ((self.chunks as f32) / 64.0).ceil() as u32;
            cpass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            &fitness_buffer,
            0,
            &fitness_staging_buffer,
            0,
            (fitness.len() * size_of::<f32>()) as u64,
        );

        let index = self.queue.submit(Some(encoder.finish()));

        let buffer_slice = fitness_staging_buffer.slice(..);
        let (send, recv) = flume::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |v| send.send(v).unwrap());
        self.device
            .poll(PollType::Wait {
                submission_index: index.into(),
                timeout: None,
            })
            .unwrap();

        recv.recv_async()
            .await
            .expect("Failed to receive map result")
            .expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        fitness_staging_buffer.unmap();

        result
            .chunks(self.chunks)
            .map(|chunk| chunk.iter().sum::<f32>() / (self.image_len as f32))
            .collect()
    }
}
