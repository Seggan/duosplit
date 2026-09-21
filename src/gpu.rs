use std::rc::Rc;
use thiserror::Error;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferAsyncError, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Instance, InstanceDescriptor, Limits, MapMode, PipelineLayoutDescriptor, PollError, PollType, PowerPreference, Queue, RequestAdapterError, RequestAdapterOptions, RequestDeviceError, ShaderModuleDescriptor, ShaderSource, ShaderStages};

#[derive(Debug, Error)]
pub enum GpuContextCreationError {
    #[error(transparent)]
    RequestAdapterError(#[from] RequestAdapterError),

    #[error("Requested memory for binding {0} is too large")]
    BindingTooLarge(String),

    #[error(transparent)]
    RequestDeviceError(#[from] RequestDeviceError),
}

#[derive(Debug, Error)]
pub enum GpuExecutionError {
    #[error("Binding {0} has not had its data set")]
    DataNotSet(String),

    #[error(transparent)]
    BufferAsyncError(#[from] BufferAsyncError),

    #[error(transparent)]
    PollError(#[from] PollError)
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BufferBindingSpec {
    pub name: String,
    pub binding_type: BufferBindingType,
    pub min_size: u32,
    pub allow_readout: bool,
}

pub struct BufferBinding {
    spec: BufferBindingSpec,
    buffer: Option<Buffer>,
    staging_buffer: Option<Buffer>,
    device: Rc<Device>,
}

impl BufferBinding {
    pub fn set_data(&mut self, data: &[u8]) {
        let mut usages = match self.spec.binding_type {
            BufferBindingType::Uniform => BufferUsages::UNIFORM,
            BufferBindingType::Storage { .. } => BufferUsages::STORAGE,
        };
        if self.spec.allow_readout {
            usages |= BufferUsages::COPY_SRC;
        }

        self.buffer = Some(self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&self.spec.name),
            contents: data,
            usage: usages,
        }));
    }

    pub fn read_data(&mut self) -> Option<Vec<u8>> {
        if let Some(staging_buffer) = self.staging_buffer.take() {
            let slice = staging_buffer.slice(..);
            let data = slice.get_mapped_range();
            let result = data.iter().copied().collect::<Vec<_>>();
            drop(data);
            staging_buffer.unmap();
            return Some(result);
        }
        None
    }
}

impl Drop for BufferBinding {
    fn drop(&mut self) {
        self.read_data();
    }
}

pub struct GpuContext {
    gpu_name: String,
    device: Rc<Device>,
    queue: Queue,
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
    bindings: Vec<BufferBinding>,
}

impl GpuContext {
    pub async fn new(
        shader: &str,
        binding_specs: Vec<BufferBindingSpec>,
    ) -> Result<Self, GpuContextCreationError> {
        let instance = Instance::new(&InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance, // use best GPU
                ..Default::default()
            })
            .await?;

        // make sure min binding amount is respected
        let max_buf_size = adapter.limits().max_buffer_size;
        let max_storage_size = adapter.limits().max_storage_buffer_binding_size;
        for binding in binding_specs.iter() {
            match binding.binding_type {
                BufferBindingType::Uniform => {
                    if binding.min_size as u64 > max_buf_size {
                        return Err(GpuContextCreationError::BindingTooLarge(
                            binding.name.clone(),
                        ));
                    }
                }

                BufferBindingType::Storage { .. } => {
                    if binding.min_size > max_storage_size {
                        return Err(GpuContextCreationError::BindingTooLarge(
                            binding.name.clone(),
                        ));
                    }
                }
            }
        }

        // use max buffer size
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                required_limits: Limits {
                    max_buffer_size: adapter.limits().max_buffer_size,
                    max_storage_buffer_binding_size: adapter
                        .limits()
                        .max_storage_buffer_binding_size,
                    max_uniform_buffer_binding_size: adapter
                        .limits()
                        .max_uniform_buffer_binding_size,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await?;

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(shader.into()),
        });

        let layout_entries = binding_specs
            .iter()
            .enumerate()
            .map(|(i, binding)| BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: binding.binding_type,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main".into(),
            compilation_options: Default::default(),
            cache: None,
        });

        let device = Rc::new(device);

        let bindings = binding_specs
            .into_iter()
            .map(|spec| BufferBinding {
                spec,
                buffer: None,
                staging_buffer: None,
                device: device.clone(),
            })
            .collect();

        Ok(Self {
            gpu_name: adapter.get_info().name,
            device,
            queue,
            pipeline,
            layout,
            bindings,
        })
    }

    pub fn get_buffer_binding(&mut self, name: &str) -> Option<&mut BufferBinding> {
        for binding in self.bindings.iter_mut() {
            if binding.spec.name == name {
                return Some(binding);
            }
        }
        None
    }

    pub fn get_gpu_name(&self) -> &str {
        &self.gpu_name
    }

    pub async fn execute(
        &mut self,
        workgroup_dims: (u32, u32, u32),
    ) -> Result<(), GpuExecutionError> {
        let bind_group_entries = self
            .bindings
            .iter()
            .enumerate()
            .map(|(i, binding)| match &binding.buffer {
                None => Err(GpuExecutionError::DataNotSet(binding.spec.name.clone())),
                Some(buffer) => Ok(BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.as_entire_binding(),
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &bind_group_entries,
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
            cpass.dispatch_workgroups(workgroup_dims.0, workgroup_dims.1, workgroup_dims.2);
        }

        // setup staging buffers
        for binding in self.bindings.iter_mut() {
            if binding.spec.allow_readout {
                let buffer = binding.buffer.as_ref().unwrap(); // buffer guaranteed to be not None cause of above check
                let staging_buffer = self.device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("{}_staging", binding.spec.name)),
                    size: buffer.size(),
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, buffer.size());

                binding.staging_buffer = Some(staging_buffer);
            }
        }

        self.queue.submit(Some(encoder.finish()));

        // copy staging buffers
        let mut channels = Vec::new();
        for binding in self.bindings.iter() {
            if let Some(staging_buffer) = &binding.staging_buffer {
                let (sender, receiver) = flume::bounded(1);
                staging_buffer
                    .slice(..)
                    .map_async(MapMode::Read, move |result| sender.send(result).unwrap());
                channels.push(receiver);
            }
        }

        self.device.poll(PollType::wait_indefinitely())?;

        for receiver in channels {
            receiver.recv_async().await.unwrap()?;
        }

        Ok(())
    }
}
