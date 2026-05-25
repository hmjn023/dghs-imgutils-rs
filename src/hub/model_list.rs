//! imgutils で使用される全モデルの定義。
//!
//! 各エントリは `(repo_id, filename, repo_type, description)` の形式です。

/// モデルエントリ
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// HuggingFace リポジトリ ID（例: `"deepghs/ccip_onnx"`）
    pub repo_id: &'static str,
    /// リポジトリ内のファイルパス
    pub filename: &'static str,
    /// リポジトリの種類（`"model"` または `"dataset"`）
    pub repo_type: &'static str,
    /// モデルの説明（機能名）
    pub description: &'static str,
}

impl ModelEntry {
    const fn new(
        repo_id: &'static str,
        filename: &'static str,
        repo_type: &'static str,
        description: &'static str,
    ) -> Self {
        Self {
            repo_id,
            filename,
            repo_type,
            description,
        }
    }

    const fn model(
        repo_id: &'static str,
        filename: &'static str,
        description: &'static str,
    ) -> Self {
        Self::new(repo_id, filename, "model", description)
    }

    const fn dataset(
        repo_id: &'static str,
        filename: &'static str,
        description: &'static str,
    ) -> Self {
        Self::new(repo_id, filename, "dataset", description)
    }
}

/// imgutils で使用される全モデルの一覧。
///
/// Python の `imgutils` ライブラリの各モジュールから参照した
/// `hf_hub_download` 呼び出しをもとに列挙しています。
pub static ALL_MODELS: &[ModelEntry] = &[
    // ==========================================
    // metrics/ccip.py  ─ キャラクター対照学習特徴抽出
    // ==========================================
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-24-randaug-pruned/model_feat.onnx",
        "CCIP(default): 特徴抽出モデル",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-24-randaug-pruned/model_metrics.onnx",
        "CCIP(default): メトリクスモデル",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-24-randaug-pruned/metrics.json",
        "CCIP(default): メトリクス設定",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-24-randaug-pruned/cluster.json",
        "CCIP(default): クラスタリング設定",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-6-randaug-pruned_fp32/model_feat.onnx",
        "CCIP(fast): 特徴抽出モデル",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-6-randaug-pruned_fp32/model_metrics.onnx",
        "CCIP(fast): メトリクスモデル",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-6-randaug-pruned_fp32/metrics.json",
        "CCIP(fast): メトリクス設定",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-6-randaug-pruned_fp32/cluster.json",
        "CCIP(fast): クラスタリング設定",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-5_fp32/model_feat.onnx",
        "CCIP(tiny): 特徴抽出モデル",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-5_fp32/model_metrics.onnx",
        "CCIP(tiny): メトリクスモデル",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-5_fp32/metrics.json",
        "CCIP(tiny): メトリクス設定",
    ),
    ModelEntry::model(
        "deepghs/ccip_onnx",
        "ccip-caformer-5_fp32/cluster.json",
        "CCIP(tiny): クラスタリング設定",
    ),
    // ==========================================
    // metrics/lpips.py  ─ タクエ差分検出
    // ==========================================
    ModelEntry::model(
        "deepghs/imgutils-models",
        "lpips/lpips_feature.onnx",
        "LPIPS: 特徴抽出モデル",
    ),
    ModelEntry::model(
        "deepghs/imgutils-models",
        "lpips/lpips_diff.onnx",
        "LPIPS: 差分計算モデル",
    ),
    // ==========================================
    // metrics/aesthetic.py  ─ 美的スコア(旧)
    // ==========================================
    ModelEntry::model(
        "skytnt/anime-aesthetic",
        "model.onnx",
        "美的スコア(旧): skytnt",
    ),
    // ==========================================
    // metrics/dbaesthetic.py  ─ 美的スコア(新)
    // ─ ClassifyModel ベースのため model.onnx / meta.json / preprocess.json をダウンロード
    // ==========================================
    ModelEntry::model(
        "deepghs/anime_aesthetic",
        "swinv2pv3_v0_448_ls0.2_x/model.onnx",
        "美的スコア(新): モデル",
    ),
    ModelEntry::model(
        "deepghs/anime_aesthetic",
        "swinv2pv3_v0_448_ls0.2_x/meta.json",
        "美的スコア(新): ラベル",
    ),
    ModelEntry::model(
        "deepghs/anime_aesthetic",
        "swinv2pv3_v0_448_ls0.2_x/samples.npz",
        "美的スコア(新): サンプル分布",
    ),
    // ==========================================
    // segment/isnetis.py  ─ キャラクターセグメンテーション
    // ==========================================
    ModelEntry::model(
        "skytnt/anime-seg",
        "isnetis.onnx",
        "セグメンテーション: ISNetIS モデル",
    ),
    // ==========================================
    // edge/lineart.py  ─ エッジ検出(lineart)
    // ==========================================
    ModelEntry::model(
        "deepghs/imgutils-models",
        "lineart/lineart.onnx",
        "エッジ検出: Lineart",
    ),
    ModelEntry::model(
        "deepghs/imgutils-models",
        "lineart/lineart_coarse.onnx",
        "エッジ検出: Lineart Coarse",
    ),
    // ==========================================
    // edge/lineart_anime.py  ─ エッジ検出(lineart anime)
    // ==========================================
    ModelEntry::model(
        "deepghs/imgutils-models",
        "lineart/lineart_anime.onnx",
        "エッジ検出: Lineart Anime",
    ),
    // ==========================================
    // detect/face.py  ─ 顔検出 (YoloModel ベース)
    // ─ model.onnx / model_type.json のパターン
    // ==========================================
    ModelEntry::model(
        "deepghs/anime_face_detection",
        "face_detect_v1.4_s/model.onnx",
        "顔検出(s): YOLOv8",
    ),
    ModelEntry::model(
        "deepghs/anime_face_detection",
        "face_detect_v1.4_n/model.onnx",
        "顔検出(n): YOLOv8",
    ),
    // ==========================================
    // detect/head.py  ─ 頭部検出
    // ==========================================
    ModelEntry::model(
        "deepghs/anime_head_detection",
        "head_detect_v2.0_s/model.onnx",
        "頭部検出(default): YOLOv8",
    ),
    // ==========================================
    // detect/person.py  ─ 人物検出
    // ==========================================
    ModelEntry::model(
        "deepghs/anime_person_detection",
        "person_detect_v1.1_m/model.onnx",
        "人物検出(m): YOLOv8",
    ),
    // ==========================================
    // detect/censor.py  ─ モザイク検出
    // ==========================================
    ModelEntry::model(
        "deepghs/anime_censor_detection",
        "censor_detect_v1.0_s/model.onnx",
        "モザイク検出(s): YOLOv8",
    ),
    // ==========================================
    // detect/nudenet.py  ─ NudeNet
    // ==========================================
    ModelEntry::model("deepghs/nudenet_onnx", "320n.onnx", "NudeNet: YOLO モデル"),
    ModelEntry::model(
        "deepghs/nudenet_onnx",
        "nms-yolov8.onnx",
        "NudeNet: NMS モデル",
    ),
    // ==========================================
    // detect/text.py  ─ テキスト検出
    // ==========================================
    ModelEntry::model(
        "deepghs/text_detection",
        "dbnetpp_resnet50_fpnc_1200e_icdar2015/end2end.onnx",
        "テキスト検出: DBNet++",
    ),
    // ==========================================
    // detect/booru_yolo.py  ─ Booru YOLO
    // ==========================================
    ModelEntry::model(
        "deepghs/booru_yolo",
        "yolov8s_aa11/model.onnx",
        "Booru YOLO(default): yolov8s_aa11",
    ),
    // ==========================================
    // validate/nsfw.py  ─ NSFW 判定
    // ==========================================
    ModelEntry::model(
        "deepghs/imgutils-models",
        "nsfw/nsfwjs.onnx",
        "NSFW判定: nsfwjs",
    ),
    ModelEntry::model(
        "deepghs/imgutils-models",
        "nsfw/inception_v3.onnx",
        "NSFW判定: Inception V3",
    ),
    // ==========================================
    // validate/safe.py  ─ セーフ判定
    // ==========================================
    ModelEntry::model(
        "mf666/shit-checker",
        "mobilenet.xs.v2.onnx",
        "セーフ判定: MobileNet XS v2",
    ),
    // ==========================================
    // tagging/deepdanbooru.py  ─ DeepDanbooru タグ付け
    // ==========================================
    ModelEntry::model(
        "deepghs/imgutils-models",
        "deepdanbooru/deepdanbooru.onnx",
        "DeepDanbooru: モデル",
    ),
    ModelEntry::model(
        "deepghs/imgutils-models",
        "deepdanbooru/deepdanbooru_tags.csv",
        "DeepDanbooru: タグリスト",
    ),
    // ==========================================
    // tagging/mldanbooru.py  ─ ML-Danbooru タグ付け
    // ==========================================
    ModelEntry::model(
        "deepghs/ml-danbooru-onnx",
        "ml_caformer_m36_dec-5-97527.onnx",
        "ML-Danbooru: モデル",
    ),
    ModelEntry::model(
        "deepghs/imgutils-models",
        "mldanbooru/mldanbooru_tags.csv",
        "ML-Danbooru: タグリスト",
    ),
    // ==========================================
    // tagging/deepgelbooru.py  ─ DeepGelbooru タグ付け
    // ==========================================
    ModelEntry::model(
        "deepghs/deepgelbooru_onnx",
        "model.onnx",
        "DeepGelbooru: モデル",
    ),
    ModelEntry::model(
        "deepghs/deepgelbooru_onnx",
        "preprocessor.json",
        "DeepGelbooru: 前処理設定",
    ),
    ModelEntry::model(
        "deepghs/deepgelbooru_onnx",
        "tags.csv",
        "DeepGelbooru: タグリスト",
    ),
    // ==========================================
    // tagging/wd14.py  ─ WD14 タガー
    // ─ 主要モデルのみ収録。selected_tags.csv は直接リポジトリから。
    // ==========================================
    ModelEntry::model(
        "deepghs/wd14_tagger_with_embeddings",
        "SmilingWolf/wd-swinv2-tagger-v3/model.onnx",
        "WD14(SwinV2_v3/default): モデル",
    ),
    ModelEntry::model(
        "deepghs/wd14_tagger_with_embeddings",
        "SmilingWolf/wd-swinv2-tagger-v3/inv.npz",
        "WD14(SwinV2_v3): 埋め込み重み",
    ),
    ModelEntry::model(
        "SmilingWolf/wd-swinv2-tagger-v3",
        "selected_tags.csv",
        "WD14(SwinV2_v3): タグリスト",
    ),
    ModelEntry::model(
        "deepghs/wd14_tagger_with_embeddings",
        "SmilingWolf/wd-eva02-large-tagger-v3/model.onnx",
        "WD14(EVA02_Large): モデル",
    ),
    ModelEntry::model(
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "selected_tags.csv",
        "WD14(EVA02_Large): タグリスト",
    ),
    // ==========================================
    // tagging/camie.py  ─ Camie タガー
    // ==========================================
    ModelEntry::model(
        "deepghs/camie_tagger_onnx",
        "initial/model.onnx",
        "Camie(initial): フルモデル",
    ),
    ModelEntry::model(
        "deepghs/camie_tagger_onnx",
        "initial/model_initial_only.onnx",
        "Camie(initial): 初期ステージのみ",
    ),
    ModelEntry::model(
        "deepghs/camie_tagger_onnx",
        "initial/selected_tags.csv",
        "Camie(initial): タグリスト",
    ),
    ModelEntry::model(
        "deepghs/camie_tagger_onnx",
        "initial/preprocess.json",
        "Camie(initial): 前処理設定",
    ),
    ModelEntry::model(
        "deepghs/camie_tagger_onnx",
        "initial/threshold.json",
        "Camie(initial): しきい値設定",
    ),
    ModelEntry::model(
        "deepghs/camie_tagger_onnx",
        "initial/initial_emb_to_pred.onnx",
        "Camie(initial): 埋め込み→予測モデル(initial)",
    ),
    ModelEntry::model(
        "deepghs/camie_tagger_onnx",
        "initial/refined_emb_to_pred.onnx",
        "Camie(initial): 埋め込み→予測モデル(refined)",
    ),
    // ==========================================
    // tagging/pixai.py  ─ PixAI タガー
    // ==========================================
    ModelEntry::model(
        "deepghs/pixai-tagger-v0.9-onnx",
        "model.onnx",
        "PixAI(v0.9): モデル",
    ),
    ModelEntry::model(
        "deepghs/pixai-tagger-v0.9-onnx",
        "selected_tags.csv",
        "PixAI(v0.9): タグリスト",
    ),
    ModelEntry::model(
        "deepghs/pixai-tagger-v0.9-onnx",
        "preprocess.json",
        "PixAI(v0.9): 前処理設定",
    ),
    // ==========================================
    // tagging/blacklist.py  ─ タグブラックリスト
    // ==========================================
    ModelEntry::dataset(
        "alea31415/tag_filtering",
        "blacklist_tags.txt",
        "タグブラックリスト",
    ),
    // ==========================================
    // tagging/overlap.py  ─ 重複タグフィルタ
    // ==========================================
    ModelEntry::dataset(
        "alea31415/tag_filtering",
        "overlap_tags_simplified.json",
        "重複タグ定義",
    ),
];
