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

use std::fmt;
use std::sync::Arc;

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
pub struct ArrayKV {
    data: Vec<u64>,
    offset: u64,
}

impl ArrayKV {
    pub fn data(&self) -> &[u64] {
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
        size: usize,
    ) -> Result<Option<Self>> {
        // Initialize with 0 (sentinel for not found)
        let mut data = vec![0; size];

        macro_rules! fill_data {
            ($ARR_TYPE:ty) => {{
                let arr = array.as_primitive::<$ARR_TYPE>();
                for (i, val) in arr.iter().enumerate() {
                    if let Some(val) = val {
                        let key = val as u64;
                        // Calculate index: key - offset
                        let idx = key.wrapping_sub(offset_val) as usize;
                        if idx >= data.len() {
                            // TODO: 完善报错信息
                            return internal_err!(
                                "failed build Array idx >= data.len()"
                            );
                        }

                        if data[idx] != 0 {
                            // Duplicates are not allowed, fallback to the default hash join implementation
                            return Ok(None);
                        }
                        data[idx] = (i) as u64 + 1;
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
        }))
    }

    pub fn get_matched_indices_with_limit_offset(
        &self,
        prob_side_keys: &[ArrayRef],
        batch_size: usize,
        current_offset: JoinHashMapOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Result<Option<JoinHashMapOffset>> {
        input_indices.clear();
        match_indices.clear();

        if prob_side_keys.len() != 1 {
            return internal_err!(
                "ArrayKV join expects 1 join key, but got {}",
                prob_side_keys.len()
            );
        }
        let array = &prob_side_keys[0];

        let end = (current_offset.0 + batch_size).min(array.len());

        macro_rules! lookup {
            ($ARR_TYPE:ty) => {{
                let arr = array.as_primitive::<$ARR_TYPE>();
                for prob_idx in current_offset.0..end {
                    if arr.is_null(prob_idx) {
                        continue;
                    }
                    // SAFETY: prob_idx is guaranteed to be within bounds by the loop range.
                    let prob_val = unsafe { arr.value_unchecked(prob_idx) } as u64;
                    let idx_in_build_side =
                        (prob_val.wrapping_sub(self.offset())) as usize;

                    if idx_in_build_side >= self.data().len()
                        || self.data()[idx_in_build_side] == 0
                    {
                        continue;
                    }
                    match_indices.push(self.data()[idx_in_build_side] - 1);
                    input_indices.push(prob_idx as u32);
                }
            }};
        }

        match array.data_type() {
            DataType::Int8 => lookup!(Int8Type),
            DataType::Int16 => lookup!(Int16Type),
            DataType::Int32 => lookup!(Int32Type),
            DataType::Int64 => lookup!(Int64Type),
            DataType::UInt8 => lookup!(UInt8Type),
            DataType::UInt16 => lookup!(UInt16Type),
            DataType::UInt32 => lookup!(UInt32Type),
            DataType::UInt64 => lookup!(UInt64Type),
            _ => {
                return internal_err!(
                    "Unsupported type for ArrayKV lookup: {:?}",
                    array.data_type()
                );
            }
        }

        if end == array.len() {
            Ok(None)
        } else {
            Ok(Some((end, None)))
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
