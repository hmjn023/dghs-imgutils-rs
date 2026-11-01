import { createHash } from 'node:crypto';
import { readFile, rename, rm, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const { version } = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'));
const targets = {
  'linux-x64': 'linux-x64-gnu',
  'win32-x64': 'win32-x64-msvc',
  'darwin-x64': 'darwin-x64',
  'darwin-arm64': 'darwin-arm64',
};
const target = targets[`${process.platform}-${process.arch}`];
if (!target || (process.platform === 'linux' && !process.report?.getReport().header.glibcVersionRuntime)) {
  throw new Error(`No prebuilt binary for ${process.platform}-${process.arch} (Linux requires glibc).`);
}

const base = `https://github.com/hmjn023/dghs-imgutils-rs/releases/download/v${encodeURIComponent(version)}`;
async function download(name) {
  const response = await fetch(`${base}/${name}`);
  if (!response.ok) throw new Error(`Failed to download ${name} for v${version}: HTTP ${response.status}. Install a released tag.`);
  return Buffer.from(await response.arrayBuffer());
}

const sums = (await download('SHA256SUMS')).toString('utf8');
const files = ['index.js', 'index.d.ts', `dghs-imgutils-rs.${target}.node`];
for (const name of files) {
  const expected = sums.split('\n').map(line => line.trim().split(/\s+/)).find(parts => parts[1] === name)?.[0];
  if (!expected || !/^[a-f0-9]{64}$/.test(expected)) throw new Error(`Missing checksum for ${name}`);
  const data = await download(name);
  const actual = createHash('sha256').update(data).digest('hex');
  if (actual !== expected) throw new Error(`Checksum mismatch for ${name}`);
  const destination = join(root, name);
  const temporary = `${destination}.${process.pid}.tmp`;
  try {
    await writeFile(temporary, data);
    await rename(temporary, destination);
  } finally {
    await rm(temporary, { force: true });
  }
}
