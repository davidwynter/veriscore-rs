use unicode_segmentation::UnicodeSegmentation;

pub fn segment_sentences(text: &str) -> Vec<String> {
    text.unicode_sentences()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub struct SlidingWinCfg { pub left: usize, pub right: usize, pub qa_mode: bool }

pub fn sliding_windows(
    question: Option<&str>,
    sentences: &[String],
    cfg: SlidingWinCfg,
) -> Vec<String> {
    let mut out = Vec::with_capacity(sentences.len());
    for i in 0..sentences.len() {
        let left = i.saturating_sub(cfg.left);
        let right = (i + cfg.right + 1).min(sentences.len());
        let mut parts = Vec::new();
        if cfg.qa_mode {
            if let Some(q) = question { parts.push(format!("Question: {q}")); }
        }
        parts.push(format!("ContextL: {}", sentences[left..i].join(" ")));
        parts.push(format!("<SOS> {} <EOS>", sentences[i]));
        if i + 1 < right { parts.push(format!("ContextR: {}", sentences[i+1..right].join(" "))); }
        out.push(parts.join("\n"));
    }
    out
}
