use candle_core::{Module, Result, Tensor, Var};
use candle_nn as nn;
use candle_nn::VarBuilder;
use nn::layer_norm;

pub(crate) struct MultiHeadAttention {
    conv_q: nn::Conv1d,
    conv_k: nn::Conv1d,
    conv_v: nn::Conv1d,
    conv_o: nn::Conv1d,
    n_heads: usize,
    k_channels: usize,
}

impl MultiHeadAttention {
    pub fn load(vb: VarBuilder, config: &crate::Config) -> Result<Self> {
        let conv1d_config = nn::Conv1dConfig::default();
        let conv_q = nn::conv1d(
            config.hidden_channels,
            config.hidden_channels,
            1,
            conv1d_config,
            vb.pp("conv_q"),
        )?;
        let conv_k = nn::conv1d(
            config.hidden_channels,
            config.hidden_channels,
            1,
            conv1d_config,
            vb.pp("conv_k"),
        )?;
        let conv_v = nn::conv1d(
            config.hidden_channels,
            config.hidden_channels,
            1,
            conv1d_config,
            vb.pp("conv_v"),
        )?;
        let conv_o = nn::conv1d(
            config.hidden_channels,
            config.hidden_channels,
            1,
            conv1d_config,
            vb.pp("conv_o"),
        )?;
        Ok(Self {
            conv_q,
            conv_k,
            conv_v,
            conv_o,
            n_heads: config.n_heads,
            k_channels: config.hidden_channels / config.n_heads,
        })
    }

    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let (b, _, t_s) = key.dims3()?;
        let t_t = query.dims()[2];
        let query = query.reshape((b, self.n_heads, self.k_channels, t_t))?;
        let key = key.reshape((b, self.n_heads, self.k_channels, t_s))?;
        let value = value.reshape((b, self.n_heads, self.k_channels, t_s))?;
        let scores = (query / (self.k_channels as f64).sqrt())?.matmul(&key.permute((2, 1))?)?;
        let p_attn = nn::ops::softmax(&scores, 2)?;
        let output = (p_attn * value)?;
        Ok(output)
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        println!("printing attention");
        let q = self.conv_q.forward(x)?;
        println!("get query");
        let k = self.conv_k.forward(c)?;
        let v = self.conv_v.forward(c)?;

        let x = self.attention(&q, &k, &v)?;
        let x = self.conv_o.forward(&x)?;
        Ok(x)
    }
}

pub(crate) struct Ffn {
    conv_1: nn::Conv1d,
    conv_2: nn::Conv1d,
    kernel_size: usize,
}

impl Ffn {
    pub fn load(vb: VarBuilder, config: &crate::Config) -> Result<Self> {
        let conv1d_config = nn::Conv1dConfig::default();
        let conv_1 = nn::conv1d(
            config.hidden_channels,
            config.filter_channels,
            config.kernel_size,
            conv1d_config,
            vb.pp("conv_1"),
        )?;

        let conv_2 = nn::conv1d(
            config.filter_channels,
            config.hidden_channels,
            config.kernel_size,
            conv1d_config,
            vb.pp("conv_2"),
        )?;

        Ok(Self {
            conv_1,
            conv_2,
            kernel_size: config.kernel_size,
        })
    }

    pub fn forward(&self, x: &Tensor, x_mask: &Tensor) -> Result<Tensor> {
        let x = self.conv_1.forward(&self.same_padding(&(x * x_mask)?)?)?;
        let x = nn::Activation::Relu.forward(&x)?;
        let x = self.conv_2.forward(&self.same_padding(&(x * x_mask)?)?)?;
        Ok(x)
    }

    // TODO: make sure padding works
    fn same_padding(&self, x: &Tensor) -> Result<Tensor> {
        match self.kernel_size {
            1 => Ok(x.to_owned()),
            _ => {
                let pad_l = (self.kernel_size - 1) / 2;
                let pad_r = self.kernel_size / 2;
                let x = x.pad_with_zeros(1, pad_l, pad_r)?;
                Ok(x)
            }
        }
    }
}

fn forward_layer_norm(layer_norm: &nn::LayerNorm, x: &Tensor) -> Result<Tensor> {
    let x_last = x.dims()[x.dims().len() - 1];
    let x = x.permute((1, x_last))?;
    let x = layer_norm.forward(&x)?;
    Ok(x)
}
pub struct AttentionEncoder {
    attention_layers: Vec<MultiHeadAttention>,
    norm_layers_1: Vec<nn::LayerNorm>,
    norm_layers_2: Vec<nn::LayerNorm>,
    ffn_layers: Vec<Ffn>,
    n_layers: usize,
}

impl AttentionEncoder {
    pub fn load(vb: VarBuilder, config: &crate::Config) -> Result<Self> {
        let mut attn_layer_vec = Vec::with_capacity(config.n_layers);
        let mut norm_layer_1_vec = Vec::with_capacity(config.n_layers);
        let mut norm_layer_2_vec = Vec::with_capacity(config.n_layers);
        let mut ffn_layer_vec = Vec::with_capacity(config.n_layers);

        for i in 0..config.n_layers {
            attn_layer_vec.push(MultiHeadAttention::load(
                vb.pp(&format!("attn_layers.{i}")),
                config,
            )?);
            ffn_layer_vec.push(Ffn::load(vb.pp(&format!("ffn_layers.{i}")), config)?);
            let vb_norm_1 = vb.pp(&format!("norm_layers_1.{i}"));
            let gamma_1 = vb_norm_1.get(config.hidden_channels, "gamma")?;
            let beta_1 = vb_norm_1.get(config.hidden_channels, "beta")?;
            norm_layer_1_vec.push(nn::LayerNorm::new(gamma_1, beta_1, 1e5));
            let vb_norm_2 = vb.pp(&format!("norm_layers_2.{i}"));
            let gamma_2 = vb_norm_2.get(config.hidden_channels, "gamma")?;
            let beta_2 = vb_norm_2.get(config.hidden_channels, "beta")?;
            norm_layer_2_vec.push(nn::LayerNorm::new(gamma_2, beta_2, 1e5));
        }
        println!("Successfully loaded Vits Parameters!");
        Ok(Self {
            attention_layers: attn_layer_vec,
            norm_layers_1: norm_layer_1_vec,
            norm_layers_2: norm_layer_2_vec,
            ffn_layers: ffn_layer_vec,
            n_layers: config.n_layers,
        })
    }

    pub fn forward(&self, x: &Tensor, x_mask: &Tensor) -> Result<Tensor> {
        let x_dim = x.dims();
        println!("{x_dim:?}");
        let attn_mask = (x_mask.unsqueeze(2)? * x_mask.unsqueeze(x_mask.dims().len() - 1))?;
        println!("attn mask passed.");
        println!("{:?}", &x_mask.transpose(2, 1).unwrap().shape());
        let x = x.matmul(&x_mask.transpose(1, 2)?)?;
        let mut result_vec = Vec::with_capacity(1);
        println!("entering loop.");
        for i in 0..self.n_layers {
            let y = self.attention_layers[i].forward(&x, &x)?;
            println!("Passed Attention Layers! iter {i}");

            let x = forward_layer_norm(&self.norm_layers_1[i], &(&x + y)?)?;
            println!("Passed Layer Norm 1! iter {i}");
            let y = self.ffn_layers[i].forward(&x, x_mask)?;
            let x = forward_layer_norm(&self.norm_layers_2[i], &(&x + y)?)?;
            if i == self.n_layers - 1 {
                result_vec.push(x)
            }
        }
        let x = result_vec[0].matmul(x_mask)?;
        Ok(x)
    }
}
