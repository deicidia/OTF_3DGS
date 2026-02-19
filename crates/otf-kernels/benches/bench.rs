use cubecl::benchmark::run_benchmark;
use cubecl::wgpu::WgpuRuntime;
use otf_kernels::ElemwiseBenchmark;
use otf_kernels::add_kernel::add_kernel::launch_unchecked as add_launch;
use otf_kernels::mul_kernel::mul_kernel::launch_unchecked as mul_launch;

const SIZE: usize = 1_048_576;

fn main() {
    let device = Default::default();

    let benchmarks: Vec<ElemwiseBenchmark<WgpuRuntime>> = vec![
        ElemwiseBenchmark::new("add", SIZE, &device, add_launch),
        ElemwiseBenchmark::new("mul", SIZE, &device, mul_launch),
    ];

    for bench in benchmarks {
        match run_benchmark(bench) {
            Ok(result) => println!("{result}"),
            Err(e) => eprintln!("Benchmark failed: {e}"),
        }
    }
}
