use std::path::PathBuf;

pub fn get_storage_dir() -> PathBuf {
    let dir = std::env::var("IU_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".cache").join("dghs-imgutils")
        });
    std::fs::create_dir_all(&dir).ok();
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_storage_dir() {
        let dir = get_storage_dir();
        assert!(dir.exists());
        assert!(dir.is_dir());
    }

    #[test]
    fn test_get_storage_dir_env() {
        let original = std::env::var("IU_HOME").ok();
        unsafe {
            std::env::set_var("IU_HOME", "/tmp/test_iu_cache");
        }
        let dir = get_storage_dir();
        assert_eq!(dir, PathBuf::from("/tmp/test_iu_cache"));
        if let Some(val) = original {
            unsafe {
                std::env::set_var("IU_HOME", val);
            }
        } else {
            unsafe {
                std::env::remove_var("IU_HOME");
            }
        }
    }
}
