//! Generate English text from an image with the public SmolVLM2 API.
//! Usage: cargo run --example smolvlm2_caption -- IMAGE [PROMPT] [MODEL_DIR]
use dghs_imgutils_rs::generic::smolvlm2::{SmolVlm2Options, smolvlm2_generate};
use std::error::Error;

/// Runs one image-conditioned prompt with automatic model download or local assets.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let image = image::open(args.get(1).ok_or("expected image path")?)?;
    let prompt = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("Describe this image in one sentence.");
    let options = SmolVlm2Options {
        model_dir: args.get(3).map(Into::into),
        ..Default::default()
    };
    println!("{}", smolvlm2_generate(&image, prompt, &options)?);
    Ok(())
}
