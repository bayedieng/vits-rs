use candle_core::Tensor;
use candle_core::pickle;
use candle_core::safetensors;
fn main() {
    let tensor_dict = safetensors::load("pretrained_ljs.safetensors", &candle_core::Device::Cpu).unwrap();

    println!("{:?}", tensor_dict)
}