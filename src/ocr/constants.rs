pub const DET_REPO_ID: &str = "deepghs/paddleocr";
pub const DET_MODEL_PATH: &str = "ch_PP-OCRv4_det/inference.onnx";
pub const DET_DESCRIPTION: &str = "PaddleOCR: Detection model (ch_PP-OCRv4)";

pub const REC_REPO_ID: &str = "deepghs/paddleocr";
pub const REC_MODEL_PATH: &str = "ch_PP-OCRv4_rec/inference.onnx";
pub const REC_DESCRIPTION: &str = "PaddleOCR: Recognition model (ch_PP-OCRv4)";

pub const CHAR_DICT_PATH: &str = "ch_PP-OCRv4_rec/ppocr_keys_v1.txt";

pub const DET_INPUT_NAME: &str = "x";
pub const DET_OUTPUT_NAME: &str = "sigmoid.0";

pub const REC_INPUT_NAME: &str = "x";
pub const REC_OUTPUT_NAME: &str = "softmax_0.tmp_0";

pub const DET_LIMIT_SIDE_LEN: u32 = 960;
pub const REC_HEIGHT: u32 = 48;
pub const REC_WIDTH: u32 = 320;

pub const DET_THRESHOLD: f32 = 0.3;
pub const DET_BOX_THRESHOLD: f32 = 0.5;
pub const DET_UNCLIP_RATIO: f32 = 1.6;
pub const DET_MAX_CANDIDATES: usize = 1000;

pub const REC_IMG_NORMALIZE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const REC_IMG_NORMALIZE_STD: [f32; 3] = [0.5, 0.5, 0.5];

pub const DET_IMG_NORMALIZE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const DET_IMG_NORMALIZE_STD: [f32; 3] = [0.229, 0.224, 0.225];
