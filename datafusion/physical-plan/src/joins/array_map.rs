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

use arrow::buffer::MutableBuffer;
use num_traits::AsPrimitive;
use std::fmt;

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
pub struct ArrayMap {
    // data[probSideVal-offset] -> valIdxInBuildSide + 1; 0 for absent
    data: Vec<u32>,
    offset: u64, // min val in buildSide
    next: Option<Vec<u32>>,
}

impl ArrayMap {
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
    ) -> Result<Self> {
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

        Ok(Self {
            data,
            offset: offset_val,
            next,
        })
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
                internal_err!(
                    "Unsupported type for ArrayKV lookup: {:?}",
                    array.data_type()
                )
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

        let arr = array.as_primitive::<T>();

        let have_null = arr.null_count() > 0;

        match &self.next {
            None => {
                for prob_idx in current_offset.0..arr.len() {
                    if build_indices.len() == limit {
                        return Ok(Some((prob_idx, None)));
                    }

                    // short circuit
                    if have_null && arr.is_null(prob_idx) {
                        continue;
                    }
                    // SAFETY: prob_idx is guaranteed to be within bounds by the loop range.
                    let prob_val = unsafe { arr.value_unchecked(prob_idx) }.as_();
                    let idx_in_build_side = (prob_val.wrapping_sub(self.offset)) as usize;

                    if idx_in_build_side >= self.data.len()
                        || self.data[idx_in_build_side] == 0
                    {
                        continue;
                    }
                    build_indices.push((self.data[idx_in_build_side] - 1) as u64);
                    probe_indices.push(prob_idx as u32);
                }
                Ok(None)
            }
            Some(next) => {
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
                            next,
                            idx,
                            next_idx as u32,
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
                    let idx_in_build_side = (prob_val.wrapping_sub(self.offset)) as usize;
                    if idx_in_build_side >= self.data.len()
                        || self.data[idx_in_build_side] == 0
                    {
                        continue;
                    }

                    let build_idx = self.data[idx_in_build_side];

                    if let Some(offset) = traverse_chain(
                        next,
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

    pub fn mark_existing_probes(
        &self,
        probe_side_keys: &[ArrayRef],
        buf: &mut MutableBuffer,
    ) -> Result<()> {
        if probe_side_keys.len() != 1 {
            return internal_err!(
                "ArrayMap join expects 1 join key, but got {}",
                probe_side_keys.len()
            );
        }
        let array = &probe_side_keys[0];

        macro_rules! fill_buffer {
            ($T:ty) => {{
                let arr = array.as_primitive::<$T>();
                for (i, val) in arr.iter().enumerate() {
                    if let Some(val) = val {
                        let key: u64 = val.as_();
                        let idx = (key.wrapping_sub(self.offset)) as usize;
                        if idx < self.data.len() && self.data[idx] != 0 {
                            arrow::util::bit_util::set_bit(buf.as_slice_mut(), i);
                        }
                    }
                }
            }};
        }

        match array.data_type() {
            DataType::Int8 => fill_buffer!(Int8Type),
            DataType::Int16 => fill_buffer!(Int16Type),
            DataType::Int32 => fill_buffer!(Int32Type),
            DataType::Int64 => fill_buffer!(Int64Type),
            DataType::UInt8 => fill_buffer!(UInt8Type),
            DataType::UInt16 => fill_buffer!(UInt16Type),
            DataType::UInt32 => fill_buffer!(UInt32Type),
            DataType::UInt64 => fill_buffer!(UInt64Type),
            _ => {
                return internal_err!(
                    "Unsupported type for ArrayMap lookup: {:?}",
                    array.data_type()
                );
            }
        }
        Ok(())
    }
}

pub enum Map {
    HashMap(Box<dyn JoinHashMapType>),
    ArrayMap(ArrayMap),
}

impl fmt::Debug for Map {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Map::HashMap(_) => write!(f, "JoinHashMap::HashMap(...)"),
            Map::ArrayMap(array_map) => f
                .debug_struct("JoinHashMap::ArrayKV")
                .field("data_len", &array_map.data.len())
                .field("offset", &array_map.offset)
                .finish(),
        }
    }
}

impl Map {
    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        match self {
            Map::HashMap(map) => map.len(),
            Map::ArrayMap(array_map) => array_map.data.len(),
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
    fn test_array_map_limit_offset_duplicate_elements() -> Result<()> {
        // Key 5: idx 0, 3, 6
        // Key 10: idx 1, 4
        // Key 15: idx 2, 5
        // Key 7: idx 7
        // Key 8: idx 8
        let build_array: ArrayRef =
            Arc::new(Int32Array::from(vec![5, 10, 15, 5, 10, 15, 5, 7, 8]));
        let offset_val = 5;
        let range = 11;

        let array_map = ArrayMap::try_new(&build_array, offset_val, range)?;

        let probe_array: ArrayRef = Arc::new(Int32Array::from(vec![5, 10, 15, 7, 8, 9]));
        let prob_side_keys = [probe_array.clone()];

        let mut prob_indices = Vec::new();
        let mut build_indices = Vec::new();
        let mut current_offset = (0, None);
        let batch_size = 2;

        // First call: Should get 2 matches for probe key 5
        let result_offset = array_map.get_matched_indices_with_limit_offset(
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

        let result_offset = array_map.get_matched_indices_with_limit_offset(
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

        let result_offset = array_map.get_matched_indices_with_limit_offset(
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

        let result_offset = array_map.get_matched_indices_with_limit_offset(
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
        let result_offset = array_map.get_matched_indices_with_limit_offset(
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
    fn test_array_map_with_limit_and_misses() -> Result<()> {
        let build_array: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let array_map = ArrayMap::try_new(&build_array, 1, 2)?;
        let probe_array: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30, 1, 2]));
        let prob_side_keys = [probe_array];

        let mut prob_indices = Vec::new();
        let mut build_indices = Vec::new();

        // batch_size=2, first call should skip 3 misses and return 2 hits
        let result_offset = array_map.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            2,
            (0, None),
            &mut prob_indices,
            &mut build_indices,
        )?;

        assert_eq!(prob_indices, vec![3, 4]);
        assert_eq!(build_indices, vec![0, 1]);
        assert!(result_offset.is_none());
        Ok(())
    }

    #[test]
    fn test_array_map_with_build_duplicates_and_misses() -> Result<()> {
        let build_array: ArrayRef = Arc::new(Int32Array::from(vec![1, 1]));
        let array_map = ArrayMap::try_new(&build_array, 1, 1)?;
        // prob: 10(m), 1(h1, h2), 20(m), 1(h1, h2)
        let probe_array: ArrayRef = Arc::new(Int32Array::from(vec![10, 1, 20, 1]));
        let prob_side_keys = [probe_array];

        let mut prob_indices = Vec::new();
        let mut build_indices = Vec::new();

        // batch_size=3, should get 2 matches from first '1' and 1 match from second '1'
        let result_offset = array_map.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            3,
            (0, None),
            &mut prob_indices,
            &mut build_indices,
        )?;

        assert_eq!(prob_indices, vec![1, 1, 3]);
        assert_eq!(build_indices, vec![0, 1, 0]);
        assert_eq!(result_offset, Some((3, Some(2))));
        Ok(())
    }
}
