// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use num_traits::AsPrimitive;
use std::fmt;
use std::sync::Arc;

use crate::joins::chain::traverse_chain;
use crate::joins::join_hash_map::JoinHashMapOffset;
use crate::joins::utils::JoinHashMapType;
use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::DataType;
use arrow::datatypes::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use datafusion_common::{Result, internal_err};

/// A "perfect" hash map for single-column integer join keys, represented as a dense array.
///
/// This structure is highly optimized for joins where the keys are integers within a limited
/// range. Instead of calculating hashes, it uses the integer key itself as an index into a
/// `Vec`, achieving O(1) lookup performance.
///
/// # NULL Handling
///
/// This optimization can be used for joins with `NullEquality::NullEqualsNothing` even if the
/// join keys contain `NULL`s. This is because:
///
/// 1. `try_new` (build side): Ignores rows with `NULL` keys when creating the map. This is
///    correct as `NULL` keys would not match anything anyway.
/// 2. `get_matched_indices_with_limit_offset` (probe side): Skips any `NULL` keys encountered
///    in the probe side input.
///
/// This structure **cannot** be used for joins with `NullEquality::NullEqualsNull` if the
/// build side contains `NULL`s, as it does not have a mechanism to store and match `NULL` values.
#[derive(Debug)]
pub struct ArrayKV {
    data: Vec<u32>,
    offset: u64,
    next: Option<Vec<u32>>,
}

impl ArrayKV {
    pub fn data(&self) -> &[u32] {
        &self.data
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Creates a new [`ArrayKV`] from the given array of join keys.
    ///
    /// Note: This function processes only the non-null values in the input `array`,
    /// effectively ignoring any rows where the key is `NULL`.
    ///
    /// TODO: Support `NullEquality::NullEqualsNull` by storing null indices in a
    /// separate `Vec` to allow for `NULL=NULL` matching in the future.
    pub(crate) fn try_new(
        array: &ArrayRef,
        offset_val: u64,
        range: usize,
    ) -> Result<Option<Self>> {
        // Initialize with 0 (sentinel for not found)
        let mut data: Vec<u32> = vec![0; range];
        let mut next: Option<Vec<u32>> = None;

        macro_rules! fill_data {
            ($ARR_TYPE:ty) => {{
                let arr = array.as_primitive::<$ARR_TYPE>();
                for (i, val) in arr.iter().enumerate().rev() {
                    if let Some(val) = val {
                        let key = val as u64;
                        // Calculate index: key - offset
                        let idx = key.wrapping_sub(offset_val) as usize;
                        if idx >= data.len() {
                            // TODO: 完善报错信息
                            return internal_err!("failed build Array idx >= data.len()");
                        }

                        if data[idx] != 0 {
                            if next.is_none() {
                                next = Some(vec![0; array.len()])
                            }
                            next.as_mut().unwrap()[i] = data[idx]
                        }
                        data[idx] = (i) as u32 + 1;
                    }
                }
            }};
        }

        match array.data_type() {
            DataType::Int8 => fill_data!(Int8Type),
            DataType::Int16 => fill_data!(Int16Type),
            DataType::Int32 => fill_data!(Int32Type),
            DataType::Int64 => fill_data!(Int64Type),
            DataType::UInt8 => fill_data!(UInt8Type),
            DataType::UInt16 => fill_data!(UInt16Type),
            DataType::UInt32 => fill_data!(UInt32Type),
            DataType::UInt64 => fill_data!(UInt64Type),
            _ => {
                return internal_err!(
                    "Unsupported type for perfect hash join conversion: {:?}",
                    array.data_type()
                );
            }
        }

        Ok(Some(Self {
            data,
            offset: offset_val,
            next,
        }))
    }

    pub fn get_matched_indices_with_limit_offset(
        &self,
        prob_side_keys: &[ArrayRef],
        limit: usize,
        current_offset: JoinHashMapOffset,
        probe_indices: &mut Vec<u32>,
        build_indices: &mut Vec<u64>,
    ) -> Result<Option<JoinHashMapOffset>> {
        if prob_side_keys.len() != 1 {
            return internal_err!(
                "ArrayKV join expects 1 join key, but got {}",
                prob_side_keys.len()
            );
        }
        let array = &prob_side_keys[0];

        match array.data_type() {
            DataType::Int8 => self.lookup_and_get_indices::<Int8Type>(
                array,
                limit,
                current_offset,
                probe_indices,
                build_indices,
            ),
            DataType::Int16 => self.lookup_and_get_indices::<Int16Type>(
                array,
                limit,
                current_offset,
                probe_indices,
                build_indices,
            ),
            DataType::Int32 => self.lookup_and_get_indices::<Int32Type>(
                array,
                limit,
                current_offset,
                probe_indices,
                build_indices,
            ),
            DataType::Int64 => self.lookup_and_get_indices::<Int64Type>(
                array,
                limit,
                current_offset,
                probe_indices,
                build_indices,
            ),
            DataType::UInt8 => self.lookup_and_get_indices::<UInt8Type>(
                array,
                limit,
                current_offset,
                probe_indices,
                build_indices,
            ),
            DataType::UInt16 => self.lookup_and_get_indices::<UInt16Type>(
                array,
                limit,
                current_offset,
                probe_indices,
                build_indices,
            ),
            DataType::UInt32 => self.lookup_and_get_indices::<UInt32Type>(
                array,
                limit,
                current_offset,
                probe_indices,
                build_indices,
            ),
            DataType::UInt64 => self.lookup_and_get_indices::<UInt64Type>(
                array,
                limit,
                current_offset,
                probe_indices,
                build_indices,
            ),
            _ => {
                return internal_err!(
                    "Unsupported type for ArrayKV lookup: {:?}",
                    array.data_type()
                );
            }
        }
    }

    fn lookup_and_get_indices<T: arrow::datatypes::ArrowNumericType>(
        &self,
        array: &ArrayRef,
        limit: usize,
        current_offset: JoinHashMapOffset,
        probe_indices: &mut Vec<u32>,
        build_indices: &mut Vec<u64>,
    ) -> Result<Option<JoinHashMapOffset>>
    where
        T::Native: Copy + AsPrimitive<u64>,
    {
        probe_indices.clear();
        build_indices.clear();
        // let (prob_cap, build_cap) = (probe_indices.capacity(), build_indices.capacity());

        let arr = array.as_primitive::<T>();

        // arr.values().get_unchecked(index)

        if self.next.is_none() {
            let end = (current_offset.0 + limit).min(arr.len());
            for prob_idx in current_offset.0..end {
                if arr.is_null(prob_idx) {
                    continue;
                }
                // SAFETY: prob_idx is guaranteed to be within bounds by the loop range.
                let prob_val = unsafe { arr.value_unchecked(prob_idx) }.as_();
                let idx_in_build_side = (prob_val.wrapping_sub(self.offset())) as usize;

                if idx_in_build_side >= self.data().len()
                    || self.data()[idx_in_build_side] == 0
                {
                    continue;
                }
                build_indices.push((self.data()[idx_in_build_side] - 1) as u64);
                probe_indices.push(prob_idx as u32);
            }
            if end == array.len() {
                Ok(None)
            } else {
                Ok(Some((end, None)))
            }
        } else {
            let mut remaining_output = limit;
            let to_skip = match current_offset {
                // None `initial_next_idx` indicates that `initial_idx` processing hasn't been started
                (idx, None) => idx,
                // Zero `initial_next_idx` indicates that `initial_idx` has been processed during
                // previous iteration, and it should be skipped
                (idx, Some(0)) => idx + 1,
                // Otherwise, process remaining `initial_idx` matches by traversing `next_chain`,
                // to start with the next index
                (idx, Some(next_idx)) => {
                    let is_last = idx == arr.len() - 1;
                    if let Some(next_offset) = traverse_chain(
                        self.next.as_ref().unwrap(),
                        idx,
                        next_idx as u32, // Cast u64 to u32
                        &mut remaining_output,
                        probe_indices,
                        build_indices,
                        is_last,
                    ) {
                        return Ok(Some(next_offset));
                    }
                    idx + 1
                }
            };

            for prob_side_idx in to_skip..arr.len() {
                if remaining_output == 0 {
                    return Ok(Some((prob_side_idx, None)));
                }

                if arr.is_null(prob_side_idx) {
                    continue;
                }

                let is_last = prob_side_idx == arr.len() - 1;

                let prob_val = unsafe { arr.value_unchecked(prob_side_idx) }.as_();
                // todo extract to func
                let idx_in_build_side = (prob_val.wrapping_sub(self.offset())) as usize;
                if idx_in_build_side >= self.data().len()
                    || self.data()[idx_in_build_side] == 0
                {
                    continue;
                }

                let build_idx = self.data()[idx_in_build_side];

                if let Some(offset) = traverse_chain(
                    self.next.as_ref().unwrap(),
                    prob_side_idx,
                    build_idx, // Pass u32 directly
                    &mut remaining_output,
                    probe_indices,
                    build_indices,
                    is_last,
                ) {
                    return Ok(Some(offset));
                }
            }
            Ok(None)
        }
    }
}

pub enum Map {
    HashMap(Arc<dyn JoinHashMapType>),
    ArrayKV(ArrayKV),
}

impl fmt::Debug for Map {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Map::HashMap(_) => write!(f, "JoinHashMap::HashMap(...)"),
            Map::ArrayKV(array_kv) => f
                .debug_struct("JoinHashMap::ArrayKV")
                .field("data_len", &array_kv.data.len())
                .field("offset", &array_kv.offset)
                .finish(),
        }
    }
}

impl Map {
    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        match self {
            Map::HashMap(map) => map.len(),
            Map::ArrayKV(array_kv) => array_kv.data.len(),
        }
    }

    /// Returns `true` if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use std::sync::Arc;

    #[test]
    fn test_array_kv_duplicate_elements() -> Result<()> {
        // Build side: values and their 0-based indices
        // Key 5: idx 0, 3, 6
        // Key 10: idx 1, 4
        // Key 15: idx 2, 5
        let build_array: ArrayRef =
            Arc::new(Int32Array::from(vec![5, 10, 15, 5, 10, 15, 5]));
        let offset_val = 0;
        let size = 20; // Max key is 15, so a size of 20 is sufficient

        let array_kv = ArrayKV::try_new(&build_array, offset_val, size)?
            .expect("should create ArrayKV");

        // Verify the internal state for next chain (FIFO order)
        // Construction iterates backwards: 6, 5, 4, 3, 2, 1, 0
        // Key 5 (indices 0, 3, 6):
        // i=6 (val 5): data[5] = 7 (idx 6+1)
        // i=3 (val 5): next[3] = 7, data[5] = 4
        // i=0 (val 5): next[0] = 4, data[5] = 1

        // So data[5] -> 1 (idx 0)
        // next[0] -> 4 (idx 3)
        // next[3] -> 7 (idx 6)
        // next[6] -> 0 (end)

        assert_eq!(array_kv.data[5], 1); // key 5 -> first index 0
        assert_eq!(array_kv.data[10], 2); // key 10 -> first index 1
        assert_eq!(array_kv.data[15], 3); // key 15 -> first index 2

        let next_chain = array_kv.next.as_ref().expect("next chain should exist");
        assert_eq!(next_chain[0], 4); // idx 0 -> next 3
        assert_eq!(next_chain[3], 7); // idx 3 -> next 6
        assert_eq!(next_chain[6], 0); // idx 6 -> end

        assert_eq!(next_chain[1], 5); // idx 1 -> next 4
        assert_eq!(next_chain[4], 0); // idx 4 -> end

        assert_eq!(next_chain[2], 6); // idx 2 -> next 5
        assert_eq!(next_chain[5], 0); // idx 5 -> end

        // Probe side
        let probe_array: ArrayRef = Arc::new(Int32Array::from(vec![5, 10, 15, 20])); // keys 5, 10, 15, 20 (not found)
        let prob_side_keys = [probe_array.clone()];

        let mut input_indices = Vec::new();
        let mut match_indices = Vec::new();
        let batch_size = 100; // Large batch size to get all results at once
        let initial_offset = (0, None);

        let result_offset = array_kv.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            batch_size,
            initial_offset,
            &mut input_indices,
            &mut match_indices,
        )?;

        assert!(result_offset.is_none());

        // Expected matches for probe_array [5, 10, 15, 20]
        // Probe index 0 (key 5) matches build indices 0, 3, 6 (FIFO)
        // Probe index 1 (key 10) matches build indices 1, 4 (FIFO)
        // Probe index 2 (key 15) matches build indices 2, 5 (FIFO)

        let expected_combined =
            vec![(0, 0), (0, 3), (0, 6), (1, 1), (1, 4), (2, 2), (2, 5)];

        let actual_combined: Vec<(u32, u64)> = input_indices
            .iter()
            .zip(match_indices.iter())
            .map(|(&p, &m)| (p, m))
            .collect();
        assert_eq!(actual_combined, expected_combined);

        Ok(())
    }

    #[test]
    fn test_array_kv_limit_offset_duplicate_elements() -> Result<()> {
        // Key 5: idx 0, 3, 6
        // Key 10: idx 1, 4
        // Key 15: idx 2, 5
        // Key 7: idx 7
        // Key 8: idx 8
        let build_array: ArrayRef =
            Arc::new(Int32Array::from(vec![5, 10, 15, 5, 10, 15, 5, 7, 8]));
        let offset_val = 5;
        let range = 11;

        let array_kv = ArrayKV::try_new(&build_array, offset_val, range)?
            .expect("should create ArrayKV");

        let probe_array: ArrayRef = Arc::new(Int32Array::from(vec![5, 10, 15, 7, 8, 9]));
        let prob_side_keys = [probe_array.clone()];

        let mut prob_indices = Vec::new();
        let mut build_indices = Vec::new();
        let mut current_offset = (0, None);
        let batch_size = 2;

        // First call: Should get 2 matches for probe key 5
        let result_offset = array_kv.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            batch_size,
            current_offset,
            &mut prob_indices,
            &mut build_indices,
        )?;

        assert_eq!(prob_indices.len(), 2);
        assert_eq!(build_indices.len(), 2);
        assert_eq!(prob_indices, vec![0, 0]);
        assert_eq!(build_indices, vec![0, 3]);

        // Offset to resume at 7 (index 6) for probe 0
        assert_eq!(result_offset, Some((0, Some(7))));

        // Second call: Should get the last match for probe key 5, then one match for key 10
        current_offset = result_offset.unwrap();

        let result_offset = array_kv.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            batch_size,
            current_offset,
            &mut prob_indices,
            &mut build_indices,
        )?;

        assert_eq!(prob_indices.len(), 2);
        assert_eq!(build_indices.len(), 2);
        assert_eq!(prob_indices, vec![0, 1]);
        assert_eq!(build_indices, vec![6, 1]); // Match 6 (probe 0), Match 1 (probe 1)

        // Offset to resume at 5 (index 4) for probe 1
        assert_eq!(result_offset, Some((1, Some(5))));

        // Third call: Should get last match for key 10, then one match for key 15
        current_offset = result_offset.unwrap();

        let result_offset = array_kv.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            batch_size,
            current_offset,
            &mut prob_indices,
            &mut build_indices,
        )?;

        assert_eq!(prob_indices.len(), 2);
        assert_eq!(build_indices.len(), 2);
        assert_eq!(prob_indices, vec![1, 2]);
        assert_eq!(build_indices, vec![4, 2]); // Match 4 (probe 1), Match 2 (probe 2)

        // Offset to resume at 6 (index 5) for probe 2
        assert_eq!(result_offset, Some((2, Some(6))));

        // Fourth call: Should get last match for key 15
        current_offset = result_offset.unwrap();

        let result_offset = array_kv.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            batch_size,
            current_offset,
            &mut prob_indices,
            &mut build_indices,
        )?;

        assert_eq!(prob_indices.len(), 2);
        assert_eq!(build_indices.len(), 2);
        assert_eq!(prob_indices, vec![2, 3]);
        assert_eq!(build_indices, vec![5, 7]); // Match 5 (probe 2)
        assert_eq!(Some((3, Some(0))), result_offset);

        current_offset = result_offset.unwrap();
        let result_offset = array_kv.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            batch_size,
            current_offset,
            &mut prob_indices,
            &mut build_indices,
        )?;

        assert_eq!(prob_indices.len(), 1);
        assert_eq!(build_indices.len(), 1);
        assert_eq!(prob_indices, vec![4]);
        assert_eq!(build_indices, vec![8]);
        assert!(result_offset.is_none());

        Ok(())
    }

    #[test]
    fn test_array_kv_with_limit_from_user() -> Result<()> {
        // buildSide: `[1, 3, 5, 7, 8, 8, 10]`
        // probSide: `[8, 10, 6, 2, 10, 4]`
        // limit: 2
        let build_array: ArrayRef =
            Arc::new(Int32Array::from(vec![1, 3, 5, 7, 8, 8, 10]));
        // min value is 1, max is 10.
        let offset_val = 1;
        let range = 10;
        let array_kv = ArrayKV::try_new(&build_array, offset_val, range)?
            .expect("should create ArrayKV");

        let probe_array: ArrayRef = Arc::new(Int32Array::from(vec![8, 10, 6, 2, 10, 4]));
        let prob_side_keys = [probe_array.clone()];

        let mut prob_indices = Vec::new();
        let mut build_indices = Vec::new();
        let mut all_prob_indices: Vec<u32> = Vec::new();
        let mut all_build_indices: Vec<u64> = Vec::new();
        let batch_size = 2;
        let mut current_offset = Some((0 as usize, None::<u64>));

        // Expected matches (probe_idx, build_idx):
        // (0, 4), (0, 5) from probe key 8
        // (1, 6) from probe key 10
        // (4, 6) from probe key 10

        // Loop until all matches are found
        while let Some(offset) = current_offset {
            let result_offset = array_kv.get_matched_indices_with_limit_offset(
                &prob_side_keys,
                batch_size,
                offset,
                &mut prob_indices,
                &mut build_indices,
            )?;
            all_prob_indices.extend(&prob_indices);
            all_build_indices.extend(&build_indices);
            current_offset = result_offset;
        }

        let expected_prob = vec![0, 0, 1, 4];
        let expected_build = vec![4, 5, 6, 6];

        assert_eq!(all_prob_indices, expected_prob);
        assert_eq!(all_build_indices, expected_build);

        Ok(())
    }
}
