use crate::tagging::tag_match::{split_to_words, words_to_matcher};
use std::collections::HashSet;

const CHAR_WHITELIST_SUFFIX: &[&str] = &[
    "anal_hair", "anal_tail", "arm_behind_head", "arm_hair",
    "arm_under_breasts", "arms_behind_head", "bird_on_head",
    "blood_in_hair", "breasts_on_glass", "breasts_on_head",
    "cat_on_head", "closed_eyes", "clothed_female_nude_female",
    "clothed_female_nude_male", "clothed_male_nude_female",
    "clothes_between_breasts", "cream_on_face", "drying_hair",
    "empty_eyes", "face_to_breasts", "facial", "food_on_face",
    "food_on_head", "game_boy", "grabbing_another_s_hair",
    "grabbing_own_breast", "gun_to_head", "half-closed_eyes",
    "head_between_breasts", "heart_in_eye", "multiple_boys",
    "multiple_girls", "object_on_breast", "object_on_head",
    "paint_splatter_on_face", "parted_lips", "penis_on_face",
    "person_on_head", "pokemon_on_head", "pubic_hair",
    "rabbit_on_head", "rice_on_face", "severed_head",
    "star_in_eye", "sticker_on_face", "tentacles_on_male",
    "tying_hair",
];

const CHAR_WHITELIST_PREFIX: &[&str] = &[
    "holding", "hand on", "hands on", "hand to", "hands to",
    "hand in", "hands in", "hand over", "hands over",
    "futa with", "futa on", "cum on", "covering", "adjusting", "rubbing",
    "sitting", "shading", "playing", "cutting",
];

const CHAR_WHITELIST_WORD: &[&str] = &["drill"];

const CHAR_SUFFIXES: &[&str] = &[
    "eyes", "skin", "hair", "bun", "bangs", "cut", "sidelocks",
    "twintails", "braid", "braids", "afro", "ahoge", "drill",
    "drills", "bald", "dreadlocks", "side up", "ponytail", "updo",
    "beard", "mustache", "pointy ears", "ear", "horn", "tail", "wing",
    "ornament", "hairband", "pupil", "bow", "eyewear", "headwear",
    "ribbon", "crown", "cap", "hat", "hairclip", "breast", "mole",
    "halo", "earrings", "animal ear fluff", "hair flower", "glasses",
    "fang", "female", "girl", "boy", "male", "beret", "heterochromia",
    "headdress", "headgear", "eyepatch", "headphones", "eyebrows", "eyelashes",
    "sunglasses", "hair intakes", "scrunchie", "ear_piercing", "head",
    "on face", "on head", "on hair", "headband", "hair rings", "under_mouth",
    "freckles", "lip", "eyeliner", "eyeshadow", "tassel", "over one eye",
    "drill", "drill hair",
];

const CHAR_PREFIXES: &[&str] = &["hair over", "hair between", "facial"];

struct WordPool {
    words: std::collections::HashMap<usize, HashSet<Vec<String>>>,
}

impl WordPool {
    fn new(words: &[&str]) -> Self {
        let mut pool = Self { words: std::collections::HashMap::new() };
        for w in words {
            pool.append(w);
        }
        pool
    }

    fn append(&mut self, text: &str) {
        for item in words_to_matcher(&split_to_words(text), true) {
            self.words.entry(item.len()).or_default().insert(item);
        }
    }

    fn contains(&self, text: &str) -> bool {
        let words = split_to_words(text);
        if let Some(set) = self.words.get(&words.len()) {
            if set.contains(&words) {
                return true;
            }
        }
        false
    }
}

struct SuffixPool {
    suffixes: std::collections::HashMap<usize, HashSet<Vec<String>>>,
}

impl SuffixPool {
    fn new(suffixes: &[&str]) -> Self {
        let mut pool = Self { suffixes: std::collections::HashMap::new() };
        for s in suffixes {
            pool.append(s);
        }
        pool
    }

    fn append(&mut self, text: &str) {
        for item in words_to_matcher(&split_to_words(text), true) {
            self.suffixes.entry(item.len()).or_default().insert(item);
        }
    }

    fn contains(&self, text: &str) -> bool {
        let words = split_to_words(text);
        for (length, set) in &self.suffixes {
            if *length > words.len() {
                continue;
            }
            let seg: Vec<String> = if *length == 0 {
                vec![]
            } else {
                words[words.len() - length..].to_vec()
            };
            if !words_to_matcher(&seg, true).is_disjoint(set) {
                return true;
            }
        }
        false
    }
}

struct PrefixPool {
    prefixes: std::collections::HashMap<usize, HashSet<Vec<String>>>,
}

impl PrefixPool {
    fn new(prefixes: &[&str]) -> Self {
        let mut pool = Self { prefixes: std::collections::HashMap::new() };
        for p in prefixes {
            pool.append(p);
        }
        pool
    }

    fn append(&mut self, text: &str) {
        for item in words_to_matcher(&split_to_words(text), false) {
            self.prefixes.entry(item.len()).or_default().insert(item);
        }
    }

    fn contains(&self, text: &str) -> bool {
        let words = split_to_words(text);
        for (length, set) in &self.prefixes {
            if *length > words.len() {
                continue;
            }
            let seg: Vec<String> = words[..*length].to_vec();
            if !words_to_matcher(&seg, false).is_disjoint(set) {
                return true;
            }
        }
        false
    }
}

pub struct CharacterTagPool {
    whitelist_suffix: SuffixPool,
    whitelist_prefix: PrefixPool,
    whitelist_words: WordPool,
    suffixes: SuffixPool,
    prefixes: PrefixPool,
}

impl CharacterTagPool {
    pub fn new() -> Self {
        Self {
            whitelist_suffix: SuffixPool::new(CHAR_WHITELIST_SUFFIX),
            whitelist_prefix: PrefixPool::new(CHAR_WHITELIST_PREFIX),
            whitelist_words: WordPool::new(CHAR_WHITELIST_WORD),
            suffixes: SuffixPool::new(CHAR_SUFFIXES),
            prefixes: PrefixPool::new(CHAR_PREFIXES),
        }
    }

    pub fn is_basic_character_tag(&self, tag: &str) -> bool {
        if self.whitelist_words.contains(tag)
            || self.whitelist_suffix.contains(tag)
            || self.whitelist_prefix.contains(tag)
        {
            return false;
        }
        self.suffixes.contains(tag) || self.prefixes.contains(tag)
    }

    pub fn drop_basic_character_tags(&self, tags: &[(String, f32)]) -> Vec<(String, f32)> {
        tags.iter()
            .filter(|(tag, _)| !self.is_basic_character_tag(tag))
            .cloned()
            .collect()
    }
}

static DEFAULT_POOL: once_cell::sync::Lazy<CharacterTagPool> =
    once_cell::sync::Lazy::new(CharacterTagPool::new);

pub fn is_basic_character_tag(tag: &str) -> bool {
    DEFAULT_POOL.is_basic_character_tag(tag)
}

pub fn drop_basic_character_tags(tags: &[(String, f32)]) -> Vec<(String, f32)> {
    DEFAULT_POOL.drop_basic_character_tags(tags)
}
