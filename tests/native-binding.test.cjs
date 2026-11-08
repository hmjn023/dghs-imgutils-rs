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
