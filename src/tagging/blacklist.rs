use crate::hub::hf_hub_download;
use crate::tagging::tag_match::{split_to_words, words_to_matcher};
use std::collections::HashSet;

fn load_online_blacklist() -> Vec<String> {
    let path = hf_hub_download(
        "alea31415/tag_filtering",
        "blacklist_tags.txt",
        Some("dataset"),
        None,
    )
    .expect("Failed to download blacklist");
    let content = std::fs::read_to_string(path).expect("Failed to read blacklist");
    content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn online_blacklist_set() -> HashSet<Vec<String>> {
    let mut set = HashSet::new();
    for tag in load_online_blacklist() {
        let words = split_to_words(&tag);
        for m in words_to_matcher(&words, true) {
            set.insert(m);
        }
    }
    set
}

fn is_blacklisted_inner(tag: &str, blacklist_set: &HashSet<Vec<String>>) -> bool {
    let words = split_to_words(tag);
    for m in words_to_matcher(&words, true) {
        if blacklist_set.contains(&m) {
            return true;
        }
    }
    false
}

pub fn is_blacklisted(tag: &str) -> bool {
    let blacklist = online_blacklist_set();
    is_blacklisted_inner(tag, &blacklist)
}

pub fn drop_blacklisted_tags(tags: &[(String, f32)], use_presets: bool) -> Vec<(String, f32)> {
    let mut blacklist = HashSet::new();
    if use_presets {
        blacklist = online_blacklist_set();
    }
    tags.iter()
        .filter(|(tag, _)| !is_blacklisted_inner(tag, &blacklist))
        .cloned()
        .collect()
}
