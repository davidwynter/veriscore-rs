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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_basic_unicode() {
        let txt = "Hello world.  Καλημέρα κόσμε!  你好。";
        let s = segment_sentences(txt);
        assert!(s.len() >= 3);
        assert_eq!(s[0], "Hello world.");
    }

    #[test]
    fn windows_with_qa_and_margins() {
        let sentences = vec![
            "A quick intro.".to_string(),
            "The core claim appears here.".to_string(),
            "Follow-up details.".to_string(),
            "Final remark.".to_string(),
        ];
        let cfg = SlidingWinCfg { left: 1, right: 1, qa_mode: true };
        let wins = sliding_windows(Some("What is claimed?"), &sentences, cfg);
        assert_eq!(wins.len(), 4);
        // Check center window has left+right context and SOS/EOS markers
        let w1 = &wins[1];
        assert!(w1.contains("Question: What is claimed?"));
        assert!(w1.contains("ContextL: A quick intro."));
        assert!(w1.contains("<SOS> The core claim appears here. <EOS>"));
        assert!(w1.contains("ContextR: Follow-up details."));
    }
}
