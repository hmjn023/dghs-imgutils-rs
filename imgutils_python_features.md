# imgutils (Python 版) 全機能・API 技術仕様書

本書は、Python 版 `dghs-imgutils` （以下 `imgutils`）が提供する全 20 個のサブモジュールについて、その機能概要、主要 API、依存関係、使用モデル、および Rust 移植における評価を詳細に調査・網羅した技術仕様書です。

今後、本プロジェクトにおいて Rust への段階的な機能移植や napi-rs を使用したネイティブバインディング設計を行う際の強力な技術基盤として活用します。

---

## 📑 目次
1. [共通基盤・設計哲学](#1-共通基盤設計哲学)
2. [全 20 モジュール詳細仕様](#2-全-20-モジュール詳細仕様)
   - [utils (共通ユーティリティ)](#utils-共通ユーティリティ)
   - [config (メタ情報設定)](#config-メタ情報設定)
   - [data (画像処理基礎データ・IO)](#data-画像処理基礎データio)
   - [preprocess (前処理)](#preprocess-前処理)
   - [generic (汎用機械学習エンジン)](#generic-汎用機械学習エンジン)
   - [detect (物体検出)](#detect-物体検出)
   - [segment (領域分割・背景透過)](#segment-領域分割背景透過)
   - [edge (線画・エッジ抽出)](#edge-線画エッジ抽出)
   - [restore (画質修復)](#restore-画質修復)
   - [upscale (超解像)](#upscale-超解像)
   - [tagging (自動タグ付け・タガー)](#tagging-自動タグ付けタガー)
   - [validate (画像検証・フィルタリング)](#validate-画像検証フィルタリング)
   - [metrics (類似度・美的メトリクス)](#metrics-類似度美的メトリクス)
   - [ocr (文字検出・認識)](#ocr-文字検出認識)
   - [pose (姿勢・関節キーポイント検出)](#pose-姿勢関節キーポイント検出)
   - [metadata (画像メタデータ・電子透かし)](#metadata-画像メタデータ電子透かし)
   - [sd (Stable Diffusion メタデータ)](#sd-stable-diffusion-メタデータ)
   - [operate (画像変形・自動検閲・切り出し)](#operate-画像変形自動検閲切り出し)
   - [resource (リソース・背景管理)](#resource-resource-背景管理)
   - [ascii (アスキーアート生成)](#ascii-アスキーアート生成)
3. [Rust 移植に向けた考察・優先度評価](#3-rust-移植に向けた考察優先度評価)

---

## 1. 共通基盤・設計哲学

`imgutils` は、アニメ・イラスト向け画像処理に特化した高速・軽量なライブラリとして設計されています。
以下の 3 つの大きな設計特徴があります：

1. **Pillow & NumPy ファースト**
   - コアの画像データ操作は OpenCV (`cv2`) に極度に依存せず、より軽量でアニメ画風の扱いに強い `Pillow (PIL)` および数値計算ライブラリ `NumPy` をベースとしています。
2. **ONNX Runtime (ORT) を推論の基盤に採用**
   - ディープラーニングモデル（YOLOv8、Swin Transformer、ResNet、NafNet など）の推論には、CPU・GPU を問わずクロスプラットフォームで高速動作する ONNX Runtime を全面的に採用しています。これにより、PyTorch 等の巨大な重量級フレームワークなしで動作可能です。
3. **オンデマンドな HuggingFace Hub 統合**
   - モデルファイルや背景リソースはライブラリ本体に同梱せず、必要な機能が呼び出されたタイミングで HuggingFace Hub から自動的にダウンロードしてローカルキャッシュ（`~/.cache/huggingface/`）へ配置するオンデマンド・ダウンロード設計となっています。

---

## 2. 全 20 モジュール詳細仕様

### utils (共通ユーティリティ)
* **概要と目的**
  - ONNX 推論セッションの管理、デバイス自動選択、スレッドセーフなキャッシュ機構、巨大画像向けの領域分割処理など、ライブラリ全体の基本ユーティリティ。
* **主要な API**
  - `open_onnx_model(path: str) -> ort.InferenceSession`
    - 指定したパスの ONNX セッションを開きます。環境（CUDA、TensorRT、CPU）を自動検出して最適な実行プロバイダをバインドします。
  - `ts_lru_cache(...)`
    - マルチスレッド競合を防ぐスレッドロック付きの LRU キャッシュデコレータ。
  - `area_batch_run(...)`
    - 超高解像度画像の推論時にメモリ不足 (OOM) を防ぐため、画像全体を小さなグリッド（タイル）に分割してバッチ推論し、後で滑らかに結合する実行エンジン。
  - `get_storage_dir() -> str`
    - キャッシュデータの保存ストレージパスを取得（環境変数 `IU_HOME` で上書き可能）。
* **使用モデル**
  - なし
* **処理の概要**
  - `threading.Lock` を用いたスレッドセーフなセッションのラップや、ONNX Runtime プロバイダの設定管理。

### config (メタ情報設定)
* **概要と目的**
  - ライブラリ全体のパッケージメタ情報（バージョン、タイトル、著者など）の静的定義。
* **主要な API**
  - `__TITLE__`, `__VERSION__` (現行 `0.19.0`), `__DESCRIPTION__`
* **使用モデル**
  - なし

### data (画像処理基礎データ・IO)
* **概要と目的**
  - 多彩な画像入力ソースのロード、画像フォーマット変換、余白パディング、レイヤー合成、URL等からの自動ダウンロードなど、画像 I/O およびピクセル基本操作。
* **主要な API**
  - `load_image(image: ImageTyping, mode=None, force_background: Optional[str] = 'white') -> Image.Image`
    - ファイルパス、バイナリ、Blob URL、一般 HTTP URL、Pillowオブジェクトから画像を透過（RGBA）を考慮しつつ読み込む万能ローダー。
  - `rgb_encode(image: ImageTyping, order_: str = 'CHW', use_float: bool = True) -> np.ndarray`
    - 画像を NumPy 配列に変換し、モデル入力用のテンソル（例: HWC/CHW）を構築。
  - `istack(*items, size=None) -> Image.Image`
    - 複数の画像や単色レイヤー、透過マスクを順に重ね合わせて 1 枚の画像に合成。
  - `pad_image_to_size(pic, size, background_color='white') -> Image.Image`
    - アスペクト比を完璧に保ったままリサイズし、余った領域を指定色でパディングする。
  - `download_image_from_url(url: str, ...)`
    - プログレスバー付きで画像をダウンロード。GitHub/HuggingFaceのURLをRAWダウンロード用に内部で動的置換。
* **使用モデル**
  - なし
* **処理の概要**
  - Pillow の画像描画機能や NumPy 配列スライス計算、`requests` や `tqdm` を用いた堅牢な I/O。

### preprocess (前処理)
* **概要と目的**
  - 動的な JSON 設定等に基づき、画像の拡大・クロップ・正規化などの前処理パイプラインをパース・構築するメタエンジン。
* **主要な API**
  - `create_pillow_transforms(config: dict)` / `parse_pillow_transforms(data: dict)`
  - `create_torchvision_transforms(config: dict)` / `parse_torchvision_transforms(data: dict)`
* **使用モデル**
  - なし

### generic (汎用機械学習エンジン)
* **概要と目的**
  - 記述されたメタデータに基づき、あらゆるモデル（YOLO、画像分類、多ラベル、CLIP/SigLIP、セグメンテーション等）の共通推論ロジックを提供する汎用バックエンド。
* **主要なクラス・API**
  - `YOLOModel`, `yolo_predict(image, ...)`
    - YOLOv8等の物体検出の予測・NMS後処理。
  - `ClassifyModel`, `classify_predict(image, ...)`
    - 画像分類（Softmax出力）とラベル解決。
  - `CLIPModel` / `SigLIPModel`
    - CLIP/SigLIP モデルを用いた画像のベクトル特徴抽出と類似度算出。
  - `YOLOSegmentationModel`
    - YOLO ベースのセグメンテーション（BBox + クラスごとのマスク抽出）。
  - `ImageEnhancer`
    - 超解像・画質修復モジュール等のために、巨大画像タイル分割処理を抽象化したフレームワーク。
* **使用モデル**
  - なし（呼び出し側で動的に指定された ONNX モデルファイルをロード）
* **処理の概要**
  - ONNX Runtime に前処理テンソルを流し、得られた出力を各ネットワークの特性（YOLOの境界ボックス、分類のロジットなど）に応じて後処理して結果を構造化する。

### detect (物体検出)
* **概要と目的**
  - アニメ画像から各種ターゲット（顔、頭部、全身、目、手、上半身、検閲部位、ヌード部位、OCRテキスト）をバウンディングボックス (BBox) として検出する。
* **主要な API**
  - `detect_faces(image, ...)` / `detect_heads(image, ...)` / `detect_person(image, ...)`
    - 顔、頭部、人物を YOLOv8 で高速検出。
  - `detect_censors(image, ...)`
    - 画像中のモザイク・検閲された領域を検出。
  - `detect_with_booru_yolo(image, model_name='yolov8s_aa11', ...)`
    - Booru-YOLO モデルを用いて 26 クラス（頭、胸、お尻、服装状態など）を検出。
  - `detect_with_nudenet(image, ...)`
    - NudeNet モデルに基づき、露出している部位を検出。
  - `detection_visualize(image, detection, ...)`
    - 検出された BBox やラベル名を元の画像上にランダムカラーで綺麗に視覚化・描画する。
* **使用モデル**
  - 顔検出: `deepghs/anime_face_detection` -> `face_detect_v1.4_s/model.onnx`
  - 頭部検出: `deepghs/anime_head_detection` -> `head_detect_v2.0_s/model.onnx`
  - 人物検出: `deepghs/anime_person_detection` -> `person_detect_v1.1_m/model.onnx`
  - モザイク検出: `deepghs/anime_censor_detection` -> `censor_detect_v1.0_s/model.onnx`
  - 目/手/上半身検出: `deepghs/anime_eye_detection`, `deepghs/anime_hand_detection`, `deepghs/anime_halfbody_detection`
  - NudeNet: `deepghs/nudenet_onnx` -> `320n.onnx`, `nms-yolov8.onnx`
  - テキスト検出: `deepghs/text_detection` -> `dbnetpp_resnet50_fpnc_1200e_icdar2015/end2end.onnx`
  - Booru YOLO: `deepghs/booru_yolo` -> `yolov8s_aa11/model.onnx`
* **処理の概要**
  - YOLO モデルの推論結果から、確率閾値によるフィルタリングと、IoU に基づく NMS（非極大値抑制）を NumPy で高速処理し、BBox とスコアを算出。

### segment (領域分割・背景透過)
* **概要と目的**
  - アニメキャラクターを背景から切り抜き、透過 PNG を作成したり、任意の背景画像と合成するセグメンテーション機能。
* **主要な API**
  - `get_isnetis_mask(image, scale=1024) -> np.ndarray`
    - 画像中のキャラクターに該当する前景マスク（白黒のアルファマップ）を取得。
  - `segment_rgba_with_isnetis(image, ...) -> Tuple[np.ndarray, Image.Image]`
    - 背景を完全に透過させた RGBA 画像オブジェクトを生成。
* **使用モデル**
  - キャラクターセグメンテーション (ISNetIS): `skytnt/anime-seg` -> `isnetis.onnx`
* **処理の概要**
  - 入力画像をリサイズ・パディングし、ISNetIS モデルで予測された境界マスクをシグモイド活性化。元の解像度にリサイズ（Bilinear補間）してアルファチャンネルに適用。

### edge (線画・エッジ抽出)
* **概要と目的**
  - イラストや写真から輪郭（線画）を抽出。非モデル系の Canny 法と、ニューラルネットワークを用いた Lineart モデルが選択可能。
* **主要な API**
  - `get_edge_by_canny(image, ...)`
    - Canny アルゴリズムによる高速・モデル不要の輪郭抽出。
  - `get_edge_by_lineart(image, coarse=False, ...)`
    - Lineart モデル（詳細、または粗め）によるイラスト輪郭の抽出。
  - `get_edge_by_lineart_anime(image, ...)`
    - アニメイラスト用に最適化された Lineart モデルによる線画生成。
* **使用モデル**
  - Lineart: `deepghs/imgutils-models` -> `lineart/lineart.onnx`, `lineart/lineart_coarse.onnx`
  - Lineart Anime: `deepghs/imgutils-models` -> `lineart/lineart_anime.onnx`
* **処理の概要**
  - Canny: OpenCV (`cv2.Canny`) に基づき、グレースケール変換とヒステリシス閾値判定でエッジを検出。
  - Lineart: 画像をテンソル化し ONNX モデルに推論させ、後処理で線画が黒・背景が白（または透過）になるよう反転処理。

### restore (画質修復)
* **概要と目的**
  - 画像のボケ、jpeg 圧縮ノイズ、ガウシアンノイズを取り除き、また画像生成 AI に対する妨害用に埋め込まれた敵対的ノイズ（Adversarial Noise）をきれいに除去・低減する。
* **主要な API**
  - `restore_with_nafnet(image, model='REDS', ...) -> Image.Image`
    - NafNet によるブレ除去・超解像。アルファチャンネル対応。
  - `restore_with_scunet(image, model='GAN', ...) -> Image.Image`
    - SCUNet による強力なデノイズ・高精度修復。
  - `remove_adversarial_noise(image, ...) -> Image.Image`
    - 敵対的ノイズをモデルなしで取り除く独自の非線形フィルタ処理。
* **使用モデル**
  - NafNet: `deepghs/image_restoration` -> `NAFNet-REDS-width64.onnx` 等
  - SCUNet: `deepghs/image_restoration` -> `SCUNet-GAN.onnx` 等
* **処理の概要**
  - `ImageEnhancer` (generic) を使い画像をタイルに分割して推論し、境界のアーティファクトを取り除きながら統合。
  - 敵対的ノイズ除去には OpenCV の `cv2.bilateralFilter`（バイラテラルフィルタ）および `cv2.ximgproc.guidedFilter`（ガイド付きフィルタ）を適用。

### upscale (超解像)
* **概要と目的**
  - アニメイラストを高解像度にアップスケーリングする超解像モジュール。
* **主要な API**
  - `upscale_with_cdc(image, model='HGSR-Mahr-Anime-SuperRes_x4', tile_size=128) -> Image.Image`
    - CDC（Component Divide and Conquer）モデルによるイラスト 4 倍超解像。
* **使用モデル**
  - CDC: `deepghs/cdc_anime_onnx` -> `HGSR-Mahr-Anime-SuperRes_x4.onnx` 等
* **処理の概要**
  - 分割・再結合（タイル推論）を全面的に活用して超解像を処理。

### tagging (自動タグ付け・タガー)
* **概要と目的**
  - アニメイラストから、キャラクター特徴、髪型、服装、背景などの Danbooru タグやレーティング、年齢制限情報を自動で予測する。
* **主要な API**
  - `get_wd14_tags(image, ...)` / `get_deepdanbooru_tags(image, ...)`
    - WD14 (SmilingWolf) や DeepDanbooru を使ったイラスト自動タグ付け。
  - `get_mldanbooru_tags(image, ...)` / `get_deepgelbooru_tags(image, ...)`
    - ML-Danbooru や DeepGelbooru を用いた予測。
  - `drop_blacklisted_tags(tags)` / `drop_overlap_tags(tags)`
    - 特定の除外タグ（ブラックリスト）の自動削除や、髪色とキャラクター固有タグが重複する際の間引き処理。
* **使用モデル**
  - WD14: `deepghs/wd14_tagger_with_embeddings` -> `SmilingWolf/wd-swinv2-tagger-v3/model.onnx` 等
  - DeepDanbooru: `deepghs/imgutils-models` -> `deepdanbooru/deepdanbooru.onnx`
  - ML-Danbooru: `deepghs/ml-danbooru-onnx` -> `ml_caformer_m36_dec-5-97527.onnx`
  - DeepGelbooru: `deepghs/deepgelbooru_onnx` -> `model.onnx`
  - Camie: `deepghs/camie_tagger_onnx` -> `initial/model.onnx`
  - PixAI: `deepghs/pixai-tagger-v0.9-onnx` -> `model.onnx`
  - タグフィルタ: `alea31415/tag_filtering` -> `blacklist_tags.txt`, `overlap_tags_simplified.json`
* **処理の概要**
  - ONNX 推論出力を Sigmoid 活性化してタグの確信度を算出。CSV タグリストからタグ名を解決し、集合演算を用いたブラックリストフィルタリングおよび重複論理処理を行う。

### validate (画像検証・フィルタリング)
* **概要と目的**
  - 画像ファイルの健全性チェック、モノクロ・カラーの数学的判定、およびイラストデータセット構築に最適なメタ分類（AI生成判定、年代画風判定など）。
* **主要な API**
  - `is_truncated(image_path) -> bool`
    - PNG / JPEG ファイルが途中で切れているかなどの物理的破損をチェック。
  - `is_monochrome(image) -> bool`
    - RGB 色空間のばらつきを解析し、モノクロ画像かを超高速に判定。
  - `check_ai(image) -> float`
    - 画像が AI によって生成されたイラストかどうかの確率を判定。
  - `check_rating(image)` / `check_style_age(image)`
    - 一般/NSFW年齢レーティング判定、およびイラストの画風年代（2000年代、2010年代、2020年代など）の推定。
  - `check_completeness(image)` / `check_furry(image)` / `check_real(image)`
    - 線画/完成画判定、ケモノ（Furry）画像判定、実写（Real）写真判定。
* **使用モデル**
  - NSFW / Safe: `deepghs/imgutils-models` -> `nsfw/nsfwjs.onnx`, `nsfw/inception_v3.onnx`
  - セーフ判定: `mf666/shit-checker` -> `mobilenet.xs.v2.onnx`
  - その他（AI判定、画風年代など）: `ClassifyModel` ベースで各用途のモデルをロード。
* **処理の概要**
  - `is_truncated`: ファイル末尾（EOF）マーカーの有無の検証、または Pillow によるデコード・ロード試行。
  - `is_monochrome`: 画像の HSV 色空間に変換し、S（彩度）および V（明度）の分布標準偏差が閾値以下であるかを高速 NumPy 解析。
  - 機械学習判定: Timms 系統の分類器 ONNX モデルを使用。

### metrics (類似度・美的メトリクス)
* **概要と目的**
  - 画像のクオリティ（美的品質）スコアリング、およびキャラクター特徴抽出（同一人物判定、立ち絵などの差分検出、および自動クラスタリング）。
* **主要な API**
  - `anime_dbaesthetic(image) -> float`
    - 最新の SwinV2 ベースのアニメ美的スコアリング (0.0 〜 1.0)。
  - `ccip_extract_feature(image)` / `ccip_difference(feat1, feat2)`
    - キャラクター画像の対照学習ベクトルを抽出し、同一人物かどうかのコサイン類似度を比較。
  - `ccip_same(image1, image2) -> bool`
    - 2つの画像のキャラクターが同一人物かを判定（閾値処理）。
  - `ccip_clustering(images: List[ImageTyping]) -> List[int]`
    - 複数の画像から同一のキャラクター同士を自動で検出し、フォルダ分けするようにグループ ID のリストを算出。
  - `lpips_extract_feature(image)` / `lpips_difference(feat1, feat2)`
    - LPIPS（Learned Perceptual Image Patch Similarity）を用いた差分（タクエ）の検出とクラスタリング。
  - `laplacian_score(image) -> float`
    - ラプラシアンフィルタの分散値による画像の「ブレ・ボケ」度合いの数値化。
* **使用モデル**
  - 美的スコア: `deepghs/anime_aesthetic` -> `swinv2pv3_v0_448_ls0.2_x/model.onnx`
  - CCIP: `deepghs/ccip_onnx` -> `ccip-caformer-24-randaug-pruned/model_feat.onnx` 等
  - LPIPS: `deepghs/imgutils-models` -> `lpips/lpips_feature.onnx`, `lpips/lpips_diff.onnx`
* **処理の概要**
  - 特徴抽出は ONNX Runtime で処理。
  - 同一キャラクターのグループ分けや差分画像の自動グルーピング（クラスタリング）には、`scikit-learn` の `DBSCAN` や `OPTICS` アルゴリズムを適用。

### ocr (文字検出・認識)
* **概要と目的**
  - 画像中の日本語、中国語、英語、韓国語などの文字の領域を検知し、高度な光学文字認識（OCR）を実行する。
* **主要な API**
  - `ocr(image) -> List[Tuple[Tuple[int,int,int,int], str, float]]`
    - 文字領域の BBox 座標、認識したテキスト文字列、および確信度スコアのリストを取得。
* **使用モデル**
  - PaddleOCR ONNX 移植版: `deepghs/paddleocr` -> `ch_PP-OCRv4_det` (検出モデル) / `ch_PP-OCRv4_rec` (認識モデル) など。
* **処理の概要**
  - 検出モデル（DBNet）でテキスト存在ヒートマップを求め、輪郭線から BBox を抽出。
  - BBox で切り出された画像（縦書きテキストは NumPy で 90 度自動回転）を認識モデル（CRNN）に流し、CTC デコードを行って文字列に変換。

### pose (姿勢・関節キーポイント検出)
* **概要と目的**
  - 人体全身の 133 箇所（顔、手、足、目など）のキーポイント関節を抽出し、スケルトンとして描画・可視化する。
* **主要な API**
  - `dwpose_estimate(image) -> List[OP18KeyPointSet]`
    - ポーズ推定した 18 キーポイントのセットリストを取得。
  - `op18_visualize(image, keypoints) -> Image.Image`
    - 関節構造ボーンをカラー描画した画像を生成。
* **使用モデル**
  - DWpose (RTMPose-x ONNX)
* **処理の概要**
  - まず人物検出を行い、人物ごとにトリミング・正規化。アフィン変換を行い、RTMPose ONNX モデルに推論させ、キーポイント座標を推定。OpenPose 18 形式に投影。

### metadata (画像メタデータ・電子透かし)
* **概要と目的**
  - PNG/JPEGの各種メタデータの読み書き、および LSB ステガノグラフィを使用した、画質を一切劣化させずに画像中に目に見えない形で情報を埋め込む電子透かし機能。
* **主要な API**
  - `read_geninfo_parameters(image) -> Optional[str]`
    - AI 画像生成パラメーターを取得。
  - `write_lsb_metadata(image, metadata: dict) -> Image.Image`
    - 画像の最下位ビット (LSB) にディクショナリデータを完全に隠し情報として埋め込む。
  - `read_lsb_metadata(image) -> dict`
    - 隠しメタデータを抽出・デコード。
* **使用モデル**
  - なし
* **処理の概要**
  - Pillow の画像ヘッダーチャンクパーサーを拡張。
  - LSB ステガノグラフィ: NumPy でピクセルデータをビットストリームにフラット展開し、各ピクセルの R/G/B 値の最下位 1 ビットを情報のバイナリストリームと書き換えるビット演算。

### sd (Stable Diffusion メタデータ)
* **概要と目的**
  - Stable Diffusion WebUI (A1111) や NovelAI などの各種生成メタデータを専門にパースし、また Safetensors などのモデルファイルのヘッダーメタデータをパース・上書きする機能。
* **主要な API**
  - `get_sdmeta_from_image(image) -> Optional[SDMetaData]`
    - 生成プロンプト、ネガティブ、シード、Sampler などの構成要素を構造体として取得。
  - `read_metadata(model_path: str) -> dict`
    - Safetensors などのモデルファイルのメタデータヘッダー部を読み込み。
* **使用モデル**
  - なし
* **処理の概要**
  - 画像の `iTXt` / `tEXt` チャンク等から得られたテキストブロックを、高度な正規表現およびパーサーモジュールを用いて解析。
  - モデルファイルのパースには、Safetensors バイナリヘッダーのバイト境界を読み取る低レベルストリーム操作を適用。

### operate (画像変形・自動検閲・切り出し)
* **概要と目的**
  - 画像のリサイズ、透過余白のスマートな自動トリミング（Squeeze）、顔や NSFW 被検物質の自動マスキング・検閲。
* **主要な API**
  - `squeeze(image) -> Image.Image`
    - 画像の外周にある「完全に透明な領域」を自動で計算し、中身のキャラクターだけを最小の BBox でトリミングして切り出す。
  - `censor_nsfw(image, method='pixelate', nipples=True, ...) -> Image.Image`
    - 物体検出 (YOLOv8) を使って露出部位（乳首・性器など）を自動検出し、`color`（単色塗りつぶし）、`blur`（ぼかし）、`pixelate`（モザイク）、`emoji`（スタンプ貼付）で検閲処理。
* **使用モデル**
  - なし（内部で `detect.censor_detect` 等に依存）
* **処理の概要**
  - Pillow のフィルタ処理 (`ImageFilter.GaussianBlur`)、`ImageDraw` の形状描画、およびスタンプの透過ブレンド合成。

### resource (resource 背景管理)
* **概要と目的**
  - 合成用・テスト用に使える高品質な背景イラストデータセットを、オンデマンドで取得・管理・ランダム取得する機能。
* **主要な API**
  - `list_bg_image_files() -> List[str]`
  - `get_bg_image(filename) -> Image.Image` / `random_bg_image() -> Image.Image`
* **使用モデル**
  - なし (HuggingFace データセット `deepghs/anime-bg` を利用)
* **処理の概要**
  - HuggingFace から `images.csv` とアーカイブファイルを `hfutils` を用いて自動ダウンロード。Pandas（または同等のインデックス解決）でレコードを検索し Pillow でロード。

### ascii (アスキーアート生成)
* **概要と目的**
  - 画像を、テキスト文字の明暗パターン（アスキーアート）に変換し、文字列またはカラーのテキスト出力としてレンダリングする。
* **主要な API**
  - `ascii_drawing(image, levels="@%#*+=-:. ") -> str`
* **使用モデル**
  - なし
* **処理の概要**
  - 画像をグレースケール変換し、設定されたフォントのアスペクト比に合わせて NumPy で補間リサイズ。各ピクセル明度を文字列配列へインデックス変換してマッピング。

---

## 3. Rust 移植に向けた考察・優先度評価

全 20 モジュールの調査結果から、Rust への移植難易度と優先度を以下のように分析・評価しました。

### 🌟 移植優先度【高】(コア基盤 & 高需要機能)
| モジュール | 移植のポイント / 活用可能な Rust Crate | 優先理由 |
|:---|:---|:---|
| **utils** / **config** | スレッドセーフなセッション管理、環境変数によるストレージパス解決。 `once_cell`, `lazy_static` などで実装。 | ライブラリ全体の共通基盤となるため最優先。 |
| **data** | 画像のロードと基本的な NumPy (ndarray) 変換。 `image` クレートおよび `ndarray` クレートを活用。 | すべての ONNX モデル入力の基礎データ構築に必須。 |
| **segment** | ISNetIS ONNX の推論と、アルファブレンド合成。 `ort` (ONNX Runtime) と `image` で記述可能。 | 背景透過キャラ切り抜きは非常に人気が高く、依存関係も少なく移植が容易。 |
| **tagging** | `ort` による多ラベル出力（Sigmoid）と、CSV タグリストのマッピング、集合演算によるフィルタリング。 | キャラクター特徴抽出やタグ付けとしての需要が最も高い主要コンポーネント。 |
| **validate** | モノクロ判定などの画像解析ロジック、および一部の軽量な Classify ONNX 推論。 | 画像破損チェックや選別フィルタリングに最適。 |

### 🌀 移植優先度【中】(追加モジュール & アルゴリズム拡張)
| モジュール | 移植のポイント / 活用可能な Rust Crate | 移植の課題 |
|:---|:---|:---|
| **detect** | YOLOv8 モデルの推論と NMS 後処理。 `ort` と `ndarray` を使用した BBox 計算。 | YOLO 座標デコードや NMS アルゴリズムの Rust 実装が必要（※すでに Python 側に NumPy 実装があるため、ロジックの直訳は容易）。 |
| **edge** | Lineart ONNX による推論、および Canny エッジ処理。 `imageproc` クレートが Canny 等を一部サポート。 | Lineart は前処理・後処理がやや多いため、Pillow との等価性の担保が重要。 |
| **metrics** | CCIP 特徴ベクトルのコサイン類似度算出、DBSCAN / OPTICS によるクラスタリング。 `linfa` (Rust機械学習クレート) で DBSCAN は対応可能。 | クラスタリングアルゴリズムの選定と結果の等価性。 |
| **metadata** / **sd** | PNG チャンクの解析、LSB 電子透かしビット演算。 `png` クレートや `exif` クレートによるバイナリ操作。 | LSB ビットストリームのデコード・エンコードの厳密なビット演算の実装。 |
| **operate** | リサイズ、パディング、検閲モザイク / ぼかし処理。 `image` および `imageproc` で描画・フィルタリング。 | 絵文字やスタンプのブレンド合成に Pillow 同等の透過レイヤー処理が必要。 |

### 🧱 移植優先度【低】(外部依存が重い、または代替手段が多い機能)
| モジュール | 移植のポイント / 活用可能な Rust Crate | 移植の課題 |
|:---|:---|:---|
| **ocr** | PaddleOCR の 2 ステージ（検出・認識）ONNX 推論。 | 縦書き時の回転、CTC デコードなど、前処理・後処理が極めて複雑。 |
| **pose** | DWpose の複数段階アフィン変換とキーポイント推定。 | ポーズデータの可視化や前処理に複雑な 2D アフィン変換の記述が必要。 |
| **restore** / **upscale** | NafNet / SCUNet / CDC 超解像。 `ImageEnhancer`（分割タイル推論）の実装。 | タイル分割してオーバーラップを考慮しながら合成する共通フレームワークが極めて高度で重い。 |
| **resource** | バックグラウンド背景の HuggingFace からの tar ダウンロードと解凍・ Pandas 代替の検索。 | tar 展開やファイル管理機能が Rust では大掛かりになる。 |
