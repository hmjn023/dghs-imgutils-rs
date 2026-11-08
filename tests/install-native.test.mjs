import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, copyFile, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

// Exercise the installer against fake Release responses without network access.
test('installs verified release assets and rejects corrupted downloads', {
  skip: process.platform !== 'linux' || process.arch !== 'x64' || !process.report.getReport().header.glibcVersionRuntime
    ? 'fixture targets Linux x64 glibc' : false,
}, async () => {
  const root = await mkdtemp(join(tmpdir(), 'imgutils-install-'));
  try {
    await mkdir(join(root, 'scripts'));
    await copyFile(new URL('../scripts/install-native.mjs', import.meta.url), join(root, 'scripts/install-native.mjs'));
    await writeFile(join(root, 'package.json'), JSON.stringify({ version: '1.2.3' }));
    await writeFile(join(root, 'mock.mjs'), `
      import { createHash } from 'node:crypto';
      const files = ['index.js', 'index.d.ts', 'dghs-imgutils-rs.linux-x64-gnu.node'];
      const contents = Object.fromEntries(files.map(name => [name, 'data for ' + name]));
      const sums = files.map(name => createHash('sha256').update(contents[name]).digest('hex') + '  ' + name).join('\\n');
      globalThis.fetch = async url => {
        if (!url.includes('/releases/download/v1.2.3/')) throw Error('wrong release');
        const name = url.split('/').pop();
        const content = name === 'SHA256SUMS' ? sums : contents[name];
        return new Response(process.env.CORRUPT && name.endsWith('.node') ? 'corrupt' : content, { status: content ? 200 : 404 });
      };
    `);
    const run = corrupt => spawnSync(process.execPath, ['--import', join(root, 'mock.mjs'), join(root, 'scripts/install-native.mjs')], {
      encoding: 'utf8', env: { ...process.env, CORRUPT: corrupt ? '1' : '' },
    });
    const success = run(false);
    assert.equal(success.status, 0, success.stderr);
    assert.equal(await readFile(join(root, 'index.js'), 'utf8'), 'data for index.js');
    const failure = run(true);
    assert.notEqual(failure.status, 0);
    assert.match(failure.stderr, /Checksum mismatch/);
    assert.equal(await readFile(join(root, 'dghs-imgutils-rs.linux-x64-gnu.node'), 'utf8'), 'data for dghs-imgutils-rs.linux-x64-gnu.node');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
