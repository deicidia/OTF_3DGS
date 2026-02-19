// kernels
pub mod add_kernel;
pub mod mul_kernel;

// benchmark
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::marker::PhantomData;

// Type alias for the kernel launch function
type LaunchFn<R> = unsafe fn(
    &ComputeClient<R>,
    CubeCount,
    CubeDim,
    TensorArg<R>,
    TensorArg<R>,
    TensorArg<R>,
) -> Result<(), LaunchError>;

// Generic benchmark for any elementwise kernel (2 inputs, 1 output)
pub struct ElemwiseBenchmark<R: Runtime> {
    name: String,
    size: usize,
    client: ComputeClient<R>,
    launcher: LaunchFn<R>,
    _r: PhantomData<R>,
}

// Implementation of the new function for ElemwiseBenchmark
impl<R: Runtime> ElemwiseBenchmark<R> {
    pub fn new(name: &str, size: usize, device: &R::Device, launcher: LaunchFn<R>) -> Self {
        Self {
            name: name.to_string(),
            size,
            client: R::client(device),
            launcher,
            _r: PhantomData,
        }
    }
}

// input of the benchmark
#[derive(Clone)]
pub struct BenchInput {
    lhs: Handle,
    rhs: Handle,
    out: Handle,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

// Implementation of the benchmark trait for ElemwiseBenchmark
impl<R: Runtime> Benchmark for ElemwiseBenchmark<R> {
    type Input = BenchInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let data: Vec<f32> = vec![512.0f32; self.size];
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * core::mem::size_of::<f32>(),
            )
        };

        let lhs = self.client.create_from_slice(bytes);
        let rhs = self.client.create_from_slice(bytes);
        let out = self.client.empty(self.size * core::mem::size_of::<f32>());

        BenchInput {
            lhs,
            rhs,
            out,
            shape: vec![self.size],
            strides: vec![1],
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static(((input.shape[0] as u32) + 255) / 256, 1, 1);

        let lhs = unsafe {
            TensorArg::from_raw_parts::<f32>(&input.lhs, &input.strides, &input.shape, 1)
        };
        let rhs = unsafe {
            TensorArg::from_raw_parts::<f32>(&input.rhs, &input.strides, &input.shape, 1)
        };
        let out = unsafe {
            TensorArg::from_raw_parts::<f32>(&input.out, &input.strides, &input.shape, 1)
        };

        unsafe {
            (self.launcher)(&self.client, cube_count, cube_dim, lhs, rhs, out)
                .map_err(|e| format!("{e:?}"))?
        };

        Ok(())
    }

    fn name(&self) -> String {
        format!("{}-{}-{}", R::name(&self.client), self.name, self.size).to_lowercase()
    }

    fn sync(&self) {
        cubecl::future::block_on(self.client.sync()).expect("Sync failed");
    }
}
