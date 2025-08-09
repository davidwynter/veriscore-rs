use crate::{serper::Serper, types::*};
use futures::{stream, StreamExt};
use anyhow::Result;
use crate::serper::Searcher;

pub async fn retrieve_for_record(serp: &dyn Searcher, rec: ExtractedClaimsRecord, concurrency: usize)
    -> Result<EvidenceRecord> {
    let tasks = rec.all_claims.iter().map(|c| {
        let s = serp;
        async move { let hits = s.search(c).await; (c.clone(), hits) }
    });

    let mut dict = Vec::with_capacity(rec.all_claims.len());
    stream::iter(tasks).buffer_unordered(concurrency).for_each(|(claim, hits)| async {
        if let Ok(items) = hits { dict.push((claim, items)); }
    }).await;

    Ok(EvidenceRecord { claims: rec, claim_snippets_dict: dict })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use crate::serper::SerperItem;

    struct FakeSearch;
    #[async_trait::async_trait]
    impl crate::serper::Searcher for FakeSearch {
        async fn search(&self, _q: &str) -> anyhow::Result<Vec<SerperItem>> {
            Ok(vec![
                SerperItem{ title:"t1".into(), link:"l1".into(), snippet:"s1".into() },
                SerperItem{ title:"t2".into(), link:"l2".into(), snippet:"s2".into() },
            ])
        }
    }

    #[tokio::test]
    async fn retrieve_attaches_results_per_claim() {
        let rec = ExtractedClaimsRecord {
            input: InputRecord { question: None, response: "r".into(), model: None, prompt_source: None },
            prompt_tok_cnt: None, response_tok_cnt: None, abstained: false,
            claim_list: vec![], all_claims: vec!["c1".into(), "c2".into()],
        };
        let ev = retrieve_for_record(&FakeSearch, rec, 8).await.unwrap();
        assert_eq!(ev.claim_snippets_dict.len(), 2);
        assert_eq!(ev.claim_snippets_dict[0].1.len(), 2);
    }
}
