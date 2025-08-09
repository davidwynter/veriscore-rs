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
