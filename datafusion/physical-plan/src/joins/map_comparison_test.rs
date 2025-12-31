#[cfg(test)]
mod tests {
    use crate::joins::array_map::ArrayMap;
    use crate::joins::join_hash_map::JoinHashMapU32;
    use crate::joins::utils::JoinHashMapType;
    use arrow::array::{ArrayRef, UInt32Array};
    use std::sync::Arc;

    #[test]
    fn test_compare_arraymap_hashmap_memory_usage() {
        let num_rows = 1_000_000;
        let densities = [1.0, 0.75, 0.5, 0.20, 0.1];
        let dup_rates = [0.0, 0.25, 0.5, 0.75];

        println!("\nMemory Comparison Matrix (num_rows = {})", num_rows);
        println!(
            "| Density | Dup Rate | ArrayMap (MB) | JoinHashMap (MB) | Ratio (AM/JHM) |"
        );
        println!(
            "|---------|----------|---------------|------------------|----------------|"
        );

        for &density in &densities {
            for &dup_rate in &dup_rates {
                let num_distinct = (num_rows as f64 * (1.0 - dup_rate)) as usize;
                // range = num_rows / density
                let range = (num_rows as f64 / density) as u32;

                // Generate distinct keys spread across the range
                let distinct_keys: Vec<u32> = (0..num_distinct)
                    .map(|i| (i as f64 * (range as f64 / num_distinct as f64)) as u32)
                    .collect();

                // Build full keys vector with duplicates by repeating distinct keys
                let mut keys = Vec::with_capacity(num_rows);
                while keys.len() < num_rows {
                    let take = (num_rows - keys.len()).min(num_distinct);
                    keys.extend_from_slice(&distinct_keys[..take]);
                }

                assert_eq!(num_rows, keys.len());

                let array: ArrayRef = Arc::new(UInt32Array::from(keys.clone()));

                let actual_min = keys.iter().min().copied().unwrap();
                let actual_max = keys.iter().max().copied().unwrap();
                let calculated_density =
                    keys.len() as f64 / (actual_max - actual_min + 1) as f64;
                let calculated_dup_rate = 1.0 - (num_distinct as f64 / keys.len() as f64);

                assert!(
                    (calculated_density - density).abs() < 0.01,
                    "Density mismatch: calculated {}, expected {}",
                    calculated_density,
                    density
                );
                assert!(
                    (calculated_dup_rate - dup_rate).abs() < 0.01,
                    "Dup rate mismatch: calculated {}, expected {}",
                    calculated_dup_rate,
                    dup_rate
                );

                // Build ArrayMap
                // ArrayMap internally uses range + 1 for data size and num_rows for next size
                let am = ArrayMap::try_new(&array, 0, range as u64).unwrap();
                assert_eq!(
                    am.num_of_distinct_key(),
                    num_distinct,
                    "ArrayMap distinct key count mismatch"
                );
                let am_mem = am.size();

                // Build JoinHashMapU32
                let mut jhm = JoinHashMapU32::with_capacity(num_rows);
                let hashes: Vec<u64> = keys.iter().map(|&k| k as u64).collect();
                let iter = hashes.iter().enumerate().map(|(i, h)| (i, h));
                jhm.update_from_iter(Box::new(iter), 0);
                assert_eq!(
                    jhm.len(),
                    num_distinct,
                    "JoinHashMap distinct key count mismatch"
                );

                let jhm_mem = jhm.size();

                println!(
                    "| {:>6.0}% | {:>7.0}% | {:>13.2} | {:>16.2} | {:>14.2}x |",
                    density * 100.0,
                    dup_rate * 100.0,
                    am_mem as f64 / 1024.0 / 1024.0,
                    jhm_mem as f64 / 1024.0 / 1024.0,
                    am_mem as f64 / jhm_mem as f64
                );
            }
        }
    }
}
