/// `12340` → `12,340`.
pub fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `14445` → `14.4k`, `2_500_000` → `2.5M`.
pub fn compact(n: u64) -> String {
    match n {
        0..1000 => n.to_string(),
        1000..1_000_000 => format!("{:.1}k", n as f64 / 1000.0),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

/// Percentage of the context window still free, rounded.
pub fn free_pct(used: u32, total: u32) -> u32 {
    if total == 0 {
        return 0;
    }
    let free = f64::from(total.saturating_sub(used)) * 100.0 / f64::from(total);
    free.round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_numbers() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(12_340), "12,340");
        assert_eq!(thousands(1_234_567), "1,234,567");
        assert_eq!(compact(950), "950");
        assert_eq!(compact(14_445), "14.4k");
        assert_eq!(compact(32_000), "32.0k");
        assert_eq!(compact(2_500_000), "2.5M");
        assert_eq!(free_pct(14_445, 32_000), 55);
        assert_eq!(free_pct(40_000, 32_000), 0);
        assert_eq!(free_pct(1, 0), 0);
    }
}
