pub mod attention;
pub mod model;

pub struct Config {
    pub inter_channels: usize,
    pub hidden_channels: usize,
    pub filter_channels: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub kernel_size: usize,
    pub resblock: usize,
    pub resblock_kernel_sizes: [usize; 3],
    pub resblock_dilation_sizes: [usize; 3],
    pub upsample_rates: [usize; 4],
    pub upsample_initial_channel: usize,
    pub upsample_kernel_sizes: [usize; 4],
    pub n_layers_q: usize,
    pub use_spectral_norm: bool,
}

impl Config {
    pub fn ljs_base() -> Self {
        Self {
            inter_channels: 192,
            hidden_channels: 192,
            filter_channels: 768,
            n_heads: 2,
            n_layers: 6,
            kernel_size: 3,
            resblock: 1,
            resblock_kernel_sizes: [3, 7, 11],
            resblock_dilation_sizes: [1, 3, 5],
            upsample_rates: [8, 8, 2, 2],
            upsample_initial_channel: 512,
            upsample_kernel_sizes: [16, 16, 4, 4],
            n_layers_q: 3,
            use_spectral_norm: false,
        }
    }
}
