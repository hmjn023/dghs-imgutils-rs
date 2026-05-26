//! 物体検出モジュールの共通型定義

/// 境界ボックス `(x1, y1, x2, y2)`。座標値はピクセル単位。
pub type BBox = (u32, u32, u32, u32);

/// 検出結果
#[derive(Debug, Clone)]
pub struct Detection {
    /// 境界ボックス
    pub bbox: BBox,
    /// クラスラベル名
    pub label: String,
    /// 信頼度スコア (0.0 .. 1.0)
    pub score: f32,
    /// オプションのインスタンスセグメンテーションマスク (H, W)
    pub mask: Option<ndarray::Array2<f32>>,
}
