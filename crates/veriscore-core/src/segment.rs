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
    fn segment_basic_unicode() {
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

    #[test]
    fn segment_sentences_basic() {
        let txt = "Hello world. This is a test! Another sentence?";
        let s = segment_sentences(txt);
        assert!(s.len() >= 3);
        assert_eq!(s[0], "Hello world.");
    }

    #[test]
    fn sliding_windows_without_question() {
        let sentences = vec![
            "Sentence one.".to_string(),
            "Sentence two.".to_string(),
            "Sentence three.".to_string(),
        ];

        let wins = sliding_windows(
            None,
            &sentences,
            SlidingWinCfg {
                left: 1,
                right: 1,
                qa_mode: false,
            },
        );

        assert_eq!(wins.len(), 3);
        assert!(wins[1].contains("ContextL: Sentence one."));
        assert!(wins[1].contains("<SOS> Sentence two. <EOS>"));
        assert!(wins[1].contains("ContextR: Sentence three."));
        assert!(!wins[1].contains("Question:"));
    }

    #[test]
    fn sliding_windows_with_question() {
        let sentences = vec![
            "First statement.".to_string(),
            "Second statement.".to_string(),
            "Third statement.".to_string(),
        ];

        let wins = sliding_windows(
            Some("What is the claim?"),
            &sentences,
            SlidingWinCfg {
                left: 1,
                right: 1,
                qa_mode: true,
            },
        );

        assert_eq!(wins.len(), 3);
        assert!(wins[1].contains("Question: What is the claim?"));
        assert!(wins[1].contains("<SOS> Second statement. <EOS>"));
    }

    #[test]
    fn sliding_windows_handles_edges() {
        let sentences = vec![
            "A.".to_string(),
            "B.".to_string(),
            "C.".to_string(),
        ];

        let wins = sliding_windows(
            None,
            &sentences,
            SlidingWinCfg {
                left: 2,
                right: 2,
                qa_mode: false,
            },
        );

        assert_eq!(wins.len(), 3);
        assert!(wins[0].contains("<SOS> A. <EOS>"));
        assert!(wins[0].contains("ContextR: B. C."));
        assert!(wins[2].contains("<SOS> C. <EOS>"));
    }
}
