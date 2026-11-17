const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtempSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const addon = require('..');

test('package entry point loads the native addon for this version', () => {
  assert.equal(addon.getVersion(), require('../package.json').version);
});

test('RGB values cross the native boundary in the requested channel order', t => {
  const dir = mkdtempSync(join(tmpdir(), 'imgutils-node-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const path = join(dir, 'pixels.png');
  // Two pixels: red then green, supplied as separate R/G/B planes.
  addon.rgbDecode([1, 0, 0, 1, 0, 0], 3, 1, 2, 'CHW', path);
  assert.deepEqual(addon.rgbEncode(path, 'HWC', false), [255, 0, 0, 0, 255, 0]);
});

test('a tensor shape mismatch reaches JavaScript as an error', () => {
  assert.throws(() => addon.rgbDecode([], 3, 1, 2, 'CHW', 'unused.png'), {
    code: 'InvalidArg',
    message: /Invalid tensor shape/,
  });
});

test('offloadInferenceSessions can be awaited repeatedly and preserves the provider settings', async () => {
  const configuration = addon.getInferenceConfiguration();
  await Promise.all([addon.offloadInferenceSessions(), addon.offloadInferenceSessions()]);
  assert.deepEqual(addon.getInferenceConfiguration(), configuration);
});

test('SmolVLM2 exposes input errors through the native boundary', t => {
  const dir = mkdtempSync(join(tmpdir(), 'imgutils-smolvlm2-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const path = join(dir, 'image.png');
  addon.rgbDecode([1, 0, 0], 3, 1, 1, 'CHW', path);
  assert.throws(() => addon.smolvlm2Generate(path, ' '), { code: 'InvalidArg', message: /prompt/ });
  assert.throws(() => addon.smolvlm2Generate(path, 'Describe', { maxNewTokens: 0 }), {
    code: 'InvalidArg', message: /max_new_tokens/,
  });
});

test('SmolVLM2 generates English from a cat image and follows the prompt', {
  skip: !process.env.DGHS_SMOLVLM2_TEST_IMAGE,
}, () => {
  const options = { modelDir: process.env.DGHS_SMOLVLM2_MODEL_DIR, maxNewTokens: 80 };
  const inference = { provider: 'cpu' };
  const image = process.env.DGHS_SMOLVLM2_TEST_IMAGE;
  const caption = addon.smolvlm2Generate(image, 'Describe this image in one sentence.', options, inference);
  console.log(caption);
  assert.match(caption.toLowerCase(), /cat/);
  assert.match(caption.toLowerCase(), /box/);
  const animal = addon.smolvlm2Generate(image, 'What animal is in the image? Answer with one word.', options, inference);
  console.log(animal);
  assert.match(animal.toLowerCase(), /cat/);
  assert.equal(addon.smolvlm2Generate(image, 'Describe this image in one sentence.', options, inference), caption);
});

test('PLaMo translation validates input before model download', () => {
  for (const text of ['', ' \n\t', '<|plamo:tag|>output']) {
    assert.throws(() => addon.plamo3Translate(text), { code: 'InvalidArg' });
  }
  assert.throws(() => addon.plamo3Translate('Hello', { maxNewTokens: 0 }), {
    code: 'InvalidArg', message: /max_new_tokens/,
  });
});

test('PLaMo translates English to Japanese and recovers after a generation limit', {
  skip: !process.env.DGHS_PLAMO3_MODEL_DIR,
}, () => {
  const options = { modelDir: process.env.DGHS_PLAMO3_MODEL_DIR, maxNewTokens: 128 };
  const inference = { provider: 'cpu' };
  const source = 'The image depicts a cardboard box with a cat inside it.';
  const translated = addon.plamo3Translate(source, options, inference);
  console.log(translated);
  assert.match(translated, /猫/);
  assert.match(translated, /箱/);
  assert.doesNotMatch(translated, /<\|plamo:/);
  const meeting = addon.plamo3Translate('The meeting has been moved to next Monday.', options, inference);
  assert.match(meeting, /来週/);
  assert.match(meeting, /月曜/);
  assert.throws(() => addon.plamo3Translate(source, { ...options, maxNewTokens: 1 }, inference), {
    code: 'GenericFailure', message: /max_new_tokens/,
  });
  assert.equal(addon.plamo3Translate(source, options, inference), translated);
});
