use crate::hub::hf_hub_download;
use std::collections::{HashMap, HashSet};

fn get_overlap_tags() -> HashMap<String, Vec<String>> {
    let path = hf_hub_download(
        "alea31415/tag_filtering",
        "overlap_tags_simplified.json",
        Some("dataset"),
        None,
    )
    .expect("Failed to download overlap tags");
    let content = std::fs::read_to_string(path).expect("Failed to read overlap tags");
    serde_json::from_str(&content).expect("Failed to parse overlap tags JSON")
}

pub fn drop_overlap_tags(tags: &[(String, f32)]) -> Vec<(String, f32)> {
    let overlap_dict = get_overlap_tags();
    let mut result = Vec::new();

    for (tag, score) in tags.iter() {
        let tag_under = tag.replace(' ', "_");
        let mut to_remove = false;

        if let Some(overlaps) = overlap_dict.get(&tag_under) {
            let overlap_set: HashSet<&String> = overlaps.iter().collect();
            for (other, _) in tags.iter() {
                let other_under = other.replace(' ', "_");
                if overlap_set.contains(&other_under) {
                    to_remove = true;
                    break;
                }
            }
        }

        if !to_remove {
            for (other, _) in tags.iter() {
                if tag != other && other.contains(tag.as_str()) {
                    to_remove = true;
                    break;
                }
            }
        }

        if !to_remove {
            result.push((tag.clone(), *score));
        }
    }

    result
}
