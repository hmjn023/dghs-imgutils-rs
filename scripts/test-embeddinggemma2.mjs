// After npm run build:debug:
// DGHS_EMBEDDINGGEMMA2_MODEL_DIR=/path/to/export node scripts/test-embeddinggemma2.mjs image.png
// Set DGHS_EMBEDDINGGEMMA2_OPENVINO_DIR to also compare a prepared Intel GPU export.
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const { embeddinggemma2TextEncode, embeddinggemma2ImageEncode } = require('../index.js');
const imagePath = process.argv[2];
assert.ok(imagePath, 'pass an image path');
const options = {
  modelDir: process.env.DGHS_EMBEDDINGGEMMA2_MODEL_DIR,
  dtype: 'q8',
  dimensions: 256,
  maxSoftTokens: 70,
};
const cpu = { provider: 'cpu', precision: 'fp32' };
const texts = ['猫の写真', 'A photograph of a cat'];
const text = embeddinggemma2TextEncode(texts, 'SearchQuery', options, cpu);
const image = embeddinggemma2ImageEncode([imagePath], options, cpu);
assert.equal(text.length, 2);
assert.equal(image.length, 1);
for (const vector of [...text, ...image]) {
  assert.equal(vector.length, 256);
  assert.ok(vector.every(Number.isFinite));
  assert.ok(Math.abs(Math.hypot(...vector) - 1) < 1e-5);
}
assert.throws(() => embeddinggemma2TextEncode(['test'], 'typo', options, cpu));
assert.throws(() => embeddinggemma2TextEncode(['test'], 'SearchQuery', { dimensions: 64 }, cpu));

const openvinoDir = process.env.DGHS_EMBEDDINGGEMMA2_OPENVINO_DIR;
const automatic = process.env.DGHS_EMBEDDINGGEMMA2_AUTO_GPU === '1';
if (openvinoDir || automatic) {
  if (openvinoDir) for (const component of ['model', 'vision_encoder']) {
    const wrapper = join(openvinoDir, 'onnx', `${component}_quantized.openvino.onnx`);
    assert.ok(existsSync(wrapper), `missing prepared GPU model: ${wrapper}`);
  }
  const gpuOptions = { ...options, modelDir: openvinoDir };
  const gpu = { provider: 'intel_gpu', precision: 'fp32' };
  if (openvinoDir && !existsSync(join(openvinoDir, 'onnx', 'model_quantized.onnx'))) {
    assert.throws(
      () => embeddinggemma2TextEncode(texts, 'SearchQuery', gpuOptions, cpu),
      /requires the intel_gpu provider/,
    );
  }
  const gpuText = embeddinggemma2TextEncode(texts, 'SearchQuery', gpuOptions, gpu);
  const gpuImage = embeddinggemma2ImageEncode([imagePath], gpuOptions, gpu);
  assert.equal(gpuText.length, text.length);
  assert.equal(gpuImage.length, image.length);
  for (const [label, actual, expected] of [
    ...gpuText.map((vector, i) => [`text ${i}`, vector, text[i]]),
    ['image', gpuImage[0], image[0]],
  ]) {
    assert.equal(actual.length, options.dimensions);
    assert.ok(actual.every(Number.isFinite), `${label} has non-finite values`);
    const norm = Math.hypot(...actual);
    assert.ok(Math.abs(norm - 1) < 1e-5, `${label} is not normalized`);
    const cosine = actual.reduce((sum, value, i) => sum + value * expected[i], 0)
      / (norm * Math.hypot(...expected));
    const maxError = Math.max(...actual.map((value, i) => Math.abs(value - expected[i])));
    assert.ok(cosine >= 0.9999, `${label} cosine ${cosine} < 0.9999`);
    assert.ok(maxError < 0.02, `${label} max absolute error ${maxError} >= 0.02`);
    console.log(`GPU ${label}: cosine=${cosine}, max_abs_error=${maxError}`);
  }
}
console.log('EmbeddingGemma 2 native image/text smoke test passed.');
