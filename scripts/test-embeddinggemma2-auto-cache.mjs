// Run after npm run build. Requires the Intel GPU/OpenVINO runtime and cached HF
// source models (or network access). No Python/uv/manual conversion is used.
import assert from 'node:assert/strict';
import { mkdtempSync, readdirSync, statSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
const image = process.argv[2];
assert.ok(image, 'pass an image path');
const cache = mkdtempSync(join(tmpdir(), 'embeddinggemma-native-'));
const script = fileURLToPath(new URL('./test-embeddinggemma2.mjs', import.meta.url));
const env = { ...process.env, IU_HOME: cache, PATH: '', DGHS_EMBEDDINGGEMMA2_AUTO_GPU: '1' };
delete env.DGHS_EMBEDDINGGEMMA2_MODEL_DIR;
delete env.DGHS_EMBEDDINGGEMMA2_OPENVINO_DIR;
function run() {
  const child = spawnSync(process.execPath, [script, image], { env, stdio: 'inherit' });
  assert.equal(child.status, 0, `native inference failed: ${child.error ?? child.signal}`);
}
function artifacts() {
  return readdirSync(cache, { recursive: true }).filter(p => /\.openvino-compiled\.(onnx|blob)$/.test(p))
    .sort().map(p => [p, statSync(join(cache, p)).mtimeMs]);
}
try {
  run();
  const cold = artifacts();
  assert.equal(cold.length, 4, 'both encoders must be generated automatically');
  run();
  assert.deepEqual(artifacts(), cold, 'warm start must reuse the existing cache');
  console.log('Native-only cold start and cross-process cache reuse passed.');
} finally {
  rmSync(cache, { recursive: true, force: true });
}
