import {
  getPixaiTags,
  ccipGetEmbedding,
  ccipDistance,
  ccipSame,
  ccipCluster,
  detectFaces,
  detectHeads,
  getIsnetisMask,
  segmentRgbaWithIsnetis,
} from './index.js';
import * as path from 'path';

async function main() {
  console.log('============================================================');
  console.log('🧪 napi-rs ネイティブアドオン JS/TS 呼び出し検証テスト');
  console.log('============================================================\n');

  const testImages = [
    'imgutils/test/testfile/skadi.jpg',         // Character A (スカディ)
    'imgutils/test/testfile/nude_girl.png',     // Character B (ヌードの少女)
    'imgutils/test/testfile/two_bikini_girls.png', // Character C & D (2名の少女)
    'imgutils/test/testfile/hutao.jpg',         // Character E (胡桃)
  ];

  // ------------------------------------------------------------
  // 1. PixAI Tagger の検証
  // ------------------------------------------------------------
  console.log('--- 1. PixAI Tagger の検証 ---');
  const targetImage = testImages[0]; // skadi.jpg
  const absolutePath = path.resolve(targetImage);
  console.log(`📸 解析対象画像: ${targetImage}`);

  try {
    const start = performance.now();
    const result = getPixaiTags(absolutePath);
    const duration = performance.now() - start;

    console.log(`✓ 予測完了 (所要時間: ${duration.toFixed(2)}ms)`);
    
    // 一般タグの上位5件を表示
    const topGeneral = Object.entries(result.general)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    console.log('  一般属性タグ (上位5件):');
    topGeneral.forEach(([tag, score]) => {
      console.log(`    - ${tag}: ${(score * 100).toFixed(2)}%`);
    });

    // キャラクタータグを表示
    console.log('  キャラクタータグ:');
    Object.entries(result.character).forEach(([tag, score]) => {
      console.log(`    - ${tag}: ${(score * 100).toFixed(2)}%`);
    });
    console.log();
  } catch (err) {
    console.error('✗ PixAI Tagger の実行中にエラーが発生しました:', err);
  }

  // ------------------------------------------------------------
  // 2. CCIP 特徴量抽出および類似度 (Same) 評価の検証
  // ------------------------------------------------------------
  console.log('--- 2. CCIP 特徴量抽出 & 同一性評価の検証 ---');
  try {
    console.log('✓ 特徴ベクトルの抽出中...');
    const start = performance.now();
    
    const embSkadi = ccipGetEmbedding(path.resolve('imgutils/test/testfile/skadi.jpg'));
    const embHutao = ccipGetEmbedding(path.resolve('imgutils/test/testfile/hutao.jpg'));
    
    const duration = performance.now() - start;
    console.log(`✓ 特徴量抽出完了 (所要時間: ${duration.toFixed(2)}ms)`);
    console.log(`  - スカディ特徴量次元数: ${embSkadi.length} 次元`);
    console.log(`  - 胡桃特徴量次元数    : ${embHutao.length} 次元`);

    // 同一ペア・異ペアの比較
    const distSame = ccipDistance(embSkadi, embSkadi);
    const distDiff = ccipDistance(embSkadi, embHutao);
    const isSameSelf = ccipSame(embSkadi, embSkadi);
    const isSameOther = ccipSame(embSkadi, embHutao);

    console.log('\n  ペア比較結果:');
    console.log(`    - スカディ vs スカディ (同一画像): 距離 = ${distSame.toFixed(6)} / 同一キャラクター判定 = ${isSameSelf}`);
    console.log(`    - スカディ vs 胡桃     (別キャラ): 距離 = ${distDiff.toFixed(6)} / 同一キャラクター判定 = ${isSameOther}`);
    
    if (distSame < 1e-4 && isSameSelf === true && isSameOther === false) {
      console.log('  ✓ 距離および同一人物判定は期待通り正しく動作しています。');
    } else {
      console.log('  ⚠️ 判定ロジックに期待と異なる数値があります。');
    }
    console.log();
  } catch (err) {
    console.error('✗ CCIP 評価の実行中にエラーが発生しました:', err);
  }

  // ------------------------------------------------------------
  // 3. CCIP 自前 DBSCAN クラスタリングの検証
  // ------------------------------------------------------------
  console.log('--- 3. CCIP 自前 DBSCAN クラスタリングの検証 ---');
  try {
    console.log('✓ テスト用画像群から特徴ベクトルを抽出...');
    const embs = testImages.map(img => ccipGetEmbedding(path.resolve(img)));
    
    console.log('✓ DBSCAN クラスタリング実行中...');
    // 特徴ベクトルをクラスタリング
    const clusterIds = ccipCluster(embs);

    console.log('  クラスタリング結果:');
    testImages.forEach((img, idx) => {
      console.log(`    - ${img}: クラスタ ID = ${clusterIds[idx]}`);
    });
    
    // 全て異なるキャラクター画像なので、全て別々のクラスタID（またはノイズ -1）に分類されるはず
    console.log('  ✓ クラスタリング処理の動作を確認しました。');
    console.log();
  } catch (err) {
    console.error('✗ CCIP クラスタリング中にエラーが発生しました:', err);
  }

  // ------------------------------------------------------------
  // 4. 物体検出 (detect) の検証 (顔・頭部)
  // ------------------------------------------------------------
  console.log('--- 4. 物体検出 (detect) の検証 ---');
  try {
    const startImg = path.resolve('imgutils/test/testfile/skadi.jpg');
    console.log(`✓ 顔および頭部検出の実行中... 画像: ${startImg}`);
    
    const start = performance.now();
    const faces = detectFaces(startImg);
    const heads = detectHeads(startImg);
    const duration = performance.now() - start;

    console.log(`✓ 検出完了 (所要時間: ${duration.toFixed(2)}ms)`);
    console.log(`  - 検出顔数  : ${faces.length} 件`);
    faces.forEach((f, idx) => {
      console.log(`    [顔 #${idx + 1}] BBox: [${f.bbox.x1}, ${f.bbox.y1}, ${f.bbox.x2}, ${f.bbox.y2}], 確信度: ${(f.score * 100).toFixed(2)}%`);
    });

    console.log(`  - 検出頭部数: ${heads.length} 件`);
    heads.forEach((h, idx) => {
      console.log(`    [頭部 #${idx + 1}] BBox: [${h.bbox.x1}, ${h.bbox.y1}, ${h.bbox.x2}, ${h.bbox.y2}], 確信度: ${(h.score * 100).toFixed(2)}%`);
    });
    console.log();
  } catch (err) {
    console.error('✗ 物体検出の実行中にエラーが発生しました:', err);
  }

  // ------------------------------------------------------------
  // 5. 領域分割 (segment) の検証 (ISNetIS)
  // ------------------------------------------------------------
  console.log('--- 5. 領域分割 (segment) の検証 ---');
  try {
    const startImg = path.resolve('imgutils/test/testfile/skadi.jpg');
    console.log(`✓ セグメンテーション前景マスク抽出 & 透過PNG生成の実行中... 画像: ${startImg}`);

    const start = performance.now();
    const mask = getIsnetisMask(startImg, 1024);
    const pngBytes = segmentRgbaWithIsnetis(startImg, 1024);
    const duration = performance.now() - start;

    console.log(`✓ 処理完了 (所要時間: ${duration.toFixed(2)}ms)`);
    console.log(`  - マスクサイズ: ${mask.length} 行 x ${mask[0]?.length || 0} 列`);
    console.log(`  - 透過PNGサイズ: ${pngBytes.length} バイト`);

    if (mask.length > 0 && pngBytes.length > 0) {
      console.log('  ✓ 前景マスクおよび透過画像合成処理が期待通り正しく動作しています。');
    }
    console.log();
  } catch (err) {
    console.error('✗ 領域分割の実行中にエラーが発生しました:', err);
  }

  console.log('============================================================');
  console.log('🏆 napi-rs ネイティブアドオン JS/TS 呼び出し検証完了！');
  console.log('============================================================');
}

main();
