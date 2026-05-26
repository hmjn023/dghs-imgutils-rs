use std::collections::HashSet;

pub fn split_to_words(text: &str) -> Vec<String> {
    text.split(|c: char| c == '_' || c == ' ')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn singular_form(word: &str) -> String {
    if word.ends_with('s') && word.len() > 1 && !word.ends_with("ss") {
        word[..word.len() - 1].to_string()
    } else {
        word.to_string()
    }
}

fn plural_form(word: &str) -> String {
    format!("{}s", word)
}

pub fn words_to_matcher(words: &[String], enable_forms: bool) -> HashSet<Vec<String>> {
    let mut set = HashSet::new();
    set.insert(words.to_vec());
    if enable_forms && !words.is_empty() {
        let mut sing = words.to_vec();
        if let Some(last) = sing.last_mut() {
            *last = singular_form(last);
        }
        set.insert(sing);
        let mut plur = words.to_vec();
        if let Some(last) = plur.last_mut() {
            *last = plural_form(last);
        }
        set.insert(plur);
    }
    set
}

pub fn tag_match_suffix(text: &str, suffix: &str) -> bool {
    let suffix_words = split_to_words(suffix);
    if suffix_words.is_empty() {
        return true;
    }
    let text_words = split_to_words(text);
    let start = text_words.len().saturating_sub(suffix_words.len());
    let text_suffix = &text_words[start..];
    !words_to_matcher(text_suffix, true).is_disjoint(&words_to_matcher(&suffix_words, true))
}

pub fn tag_match_prefix(text: &str, prefix: &str) -> bool {
    let prefix_words = split_to_words(prefix);
    if prefix_words.is_empty() {
        return true;
    }
    let text_words = split_to_words(text);
    let end = prefix_words.len().min(text_words.len());
    let text_prefix = &text_words[..end];
    !words_to_matcher(text_prefix, false).is_disjoint(&words_to_matcher(&prefix_words, false))
}

pub fn tag_match_full(t1: &str, t2: &str) -> bool {
    let t1_words = split_to_words(t1);
    let t2_words = split_to_words(t2);
    !words_to_matcher(&t1_words, true).is_disjoint(&words_to_matcher(&t2_words, true))
}
