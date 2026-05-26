use std::collections::HashMap;

const KAOMOJIS: &[&str] = &[
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=",
    ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
];

pub fn add_underline(tag: &str) -> String {
    tag.trim().replace(' ', "_")
}

pub fn remove_underline(tag: &str) -> String {
    let tag = tag.trim();
    if KAOMOJIS.contains(&tag) {
        tag.to_string()
    } else {
        tag.replace('_', " ")
    }
}

pub fn tags_to_text(
    tags: &HashMap<String, f32>,
    use_spaces: bool,
    use_escape: bool,
    include_score: bool,
    score_descend: bool,
) -> String {
    let mut pairs: Vec<(&String, &f32)> = tags.iter().collect();
    if score_descend {
        pairs.sort_by(|a, b| {
            b.1.partial_cmp(a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        });
    }

    let items: Vec<String> = pairs
        .iter()
        .map(|(tag, score)| {
            let mut t = (*tag).clone();
            if use_spaces {
                t = remove_underline(&t);
            }
            if use_escape {
                t = t.replace('\\', "\\\\");
                t = t.replace('(', "\\(");
                t = t.replace(')', "\\)");
            }
            if include_score {
                format!("({}:{:.3})", t, score)
            } else {
                t
            }
        })
        .collect();

    items.join(", ")
}
