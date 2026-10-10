// After npm run build:debug:
// DGHS_EMBEDDINGGEMMA2_MODEL_DIR=/path/to/export node scripts/test-embeddinggemma2.mjs image.png
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

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
const text = embeddinggemma2TextEncode(['猫の写真', 'A photograph of a cat'], 'SearchQuery', options, cpu);
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
console.log('EmbeddingGemma 2 native image/text smoke test passed.');
