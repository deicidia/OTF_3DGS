use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn mul_kernel(input1: &Tensor<f32>, input2: &Tensor<f32>, output: &mut Tensor<f32>) {
    if ABSOLUTE_POS < input1.len() {
        output[ABSOLUTE_POS] = input1[ABSOLUTE_POS] * input2[ABSOLUTE_POS];
    }
}
