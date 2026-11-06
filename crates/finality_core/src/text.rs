//! Text helpers shared by the tool search and the plan lookup table.

/// Lowercase alphanumeric words of at least two characters.
pub fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// BM25 scores (k1 = 1.2, b = 0.75) of `docs` for `query`; only positive scores, best first.
pub fn bm25(docs: &[String], query: &str) -> Vec<(usize, f64)> {
    const K1: f64 = 1.2;
    const B: f64 = 0.75;
    let docs: Vec<Vec<String>> = docs.iter().map(|d| tokens(d)).collect();
    let n = docs.len() as f64;
    let avg = docs.iter().map(Vec::len).sum::<usize>() as f64 / n.max(1.0);
    let mut query = tokens(query);
    query.sort();
    query.dedup();
    let mut scores: Vec<(usize, f64)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            let len = doc.len() as f64;
            let score = query
                .iter()
                .map(|q| {
                    let df = docs.iter().filter(|d| d.contains(q)).count() as f64;
                    let tf = doc.iter().filter(|t| *t == q).count() as f64;
                    if tf == 0.0 {
                        return 0.0;
                    }
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * len / avg.max(1.0)))
                })
                .sum::<f64>();
            (i, score)
        })
        .filter(|(_, s)| *s > 0.0)
        .collect();
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_matching_documents() {
        let docs = vec![
            "add a cli flag with clap".to_owned(),
            "deploy the web service".to_owned(),
            "fix the clap parser and add tests".to_owned(),
        ];
        let hits = bm25(&docs, "clap flag");
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits.len(), 2);
        assert!(bm25(&docs, "zzz").is_empty());
        assert_eq!(tokens("A-b CLI_flag"), ["cli", "flag"]);
    }
}
