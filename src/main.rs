use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use vits_rs::attention::AttentionEncoder;
use vits_rs::Config;
fn main() {
    let input_tensor_x = Tensor::rand(0.1, 1.0, (1, 1, 33), &Device::Cpu).unwrap();
    let input_tensor_x_mask = Tensor::rand(0.1, 1.0, (1, 192, 33), &Device::Cpu).unwrap();

    let config = Config::ljs_base();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&["vits_ljs.safetensors"], DType::F32, &Device::Cpu)
            .unwrap()
    };
    let vb = vb.pp("enc_p.encoder");
    let encoder = AttentionEncoder::load(vb, &config).unwrap();
    let out = encoder
        .forward(&input_tensor_x, &input_tensor_x_mask)
        .unwrap();
    println!("{out}");
}
