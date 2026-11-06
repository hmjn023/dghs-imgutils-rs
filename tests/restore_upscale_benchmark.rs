//! Opt-in Rust restoration benchmarks; model errors fail the test.
//! `cargo test --no-default-features --features openvino --test restore_upscale_benchmark -- --ignored --nocapture`
mod support;
use dghs_imgutils_rs::{
    restore::{adversarial, nafnet, scunet},
    upscale::cdc,
};

#[test]
#[ignore = "manual timing test"]
fn restore_benchmark_adversarial() {
    let image = support::image();
    support::benchmark("adversarial", 3, || {
        let output = adversarial::remove_adversarial_noise(&image, 4, 1)
            .expect("adversarial restoration failed");
        support::image_size(&output, image.width(), image.height());
    });
}
#[test]
#[ignore = "downloads NAFNet and requires ONNX Runtime"]
fn restore_benchmark_nafnet() {
    let image = support::image();
    support::benchmark("NAFNet REDS", 2, || {
        let output = nafnet::restore_with_nafnet(&image, "REDS").expect("NAFNet inference failed");
        support::image_size(&output, image.width(), image.height());
    });
}
#[test]
#[ignore = "downloads SCUNet and requires a supported ONNX Runtime backend"]
fn restore_benchmark_scunet() {
    let image = support::image();
    support::benchmark("SCUNet GAN", 2, || {
        let output = scunet::restore_with_scunet(&image, "GAN").expect("SCUNet inference failed");
        support::image_size(&output, image.width(), image.height());
    });
}
#[test]
#[ignore = "downloads CDC and requires ONNX Runtime"]
fn upscale_benchmark_cdc() {
    let image = support::image();
    support::benchmark("CDC x4", 1, || {
        let output = cdc::upscale_with_cdc(&image, cdc::DEFAULT_CDC_MODEL, 128, 16, 1)
            .expect("CDC inference failed");
        support::image_size(&output, image.width() * 4, image.height() * 4);
    });
}
