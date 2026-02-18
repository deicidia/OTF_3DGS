use cubecl::prelude::*;

#[cube(launch)]
pub fn hello_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
    }
}

pub fn run_example() {
    println!("Kernel execution logic would go here.");
}
