pub mod area;
pub mod cache;
pub mod storage;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum UtilsError {
    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Area batch run error: {0}")]
    Area(String),
}
