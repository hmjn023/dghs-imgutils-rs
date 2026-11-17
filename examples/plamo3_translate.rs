//! Translate English text with the public PLaMo 3 API.
//! Usage: cargo run --example plamo3_translate -- MODEL_DIR [ENGLISH_TEXT]
//! Use "-" as MODEL_DIR to download the Q4 model automatically.
use dghs_imgutils_rs::generic::plamo3::{Plamo3Options, plamo3_translate};
use std::error::Error;

/// Runs a single English-to-Japanese translation.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let model_dir = args.get(1).ok_or("expected model directory or -")?;
    let options = Plamo3Options {
        model_dir: (model_dir != "-").then(|| model_dir.into()),
        ..Default::default()
    };
    let text = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("The image depicts a cardboard box with a cat inside it.");
    println!("{}", plamo3_translate(text, &options)?);
    Ok(())
}
