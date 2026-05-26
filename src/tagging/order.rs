use std::collections::HashMap;

fn is_count_tag(tag: &str) -> bool {
    let tag = tag.trim();
    if tag == "solo" {
        return true;
    }
    let bytes = tag.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 {
        return false;
    }
    let rest = &tag[i..];
    let rest_str = &*rest;
    rest_str == "boy"
        || rest_str == "boys"
        || rest_str == "girl"
        || rest_str == "girls"
        || rest_str.starts_with('+')
            && (rest_str[1..] == *"boy"
                || rest_str[1..] == *"boys"
                || rest_str[1..] == *"girl"
                || rest_str[1..] == *"girls")
}

pub fn sort_tags(tags: &HashMap<String, f32>, mode: &str) -> Vec<String> {
    let mut npeople_tags = Vec::new();
    let mut remaining = Vec::new();

    for tag in tags.keys() {
        if is_count_tag(tag) {
            npeople_tags.push(tag.clone());
        } else {
            remaining.push(tag.clone());
        }
    }

    match mode {
        "original" => {}
        "score" => {
            remaining.sort_by(|a, b| {
                let sa = tags.get(a).copied().unwrap_or(0.0);
                let sb = tags.get(b).copied().unwrap_or(0.0);
                sb.partial_cmp(&sa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(b))
            });
        }
        _ => {}
    }

    npeople_tags.extend(remaining);
    npeople_tags
}
