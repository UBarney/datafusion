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
use arrow::array::{ArrayRef, AsArray};
use arrow::datatypes::{
    Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type,
    UInt8Type,
};
use arrow::datatypes::DataType;
use datafusion_common::{internal_err, Result};

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

    pub(crate) fn try_new(
        left_values: &[ArrayRef],
        offset_val: u64,
        size: usize,
    ) -> Result<Option<Self>> {
        // Initialize with 0 (sentinel for not found)
        let mut data = vec![0; size];

        let array = &left_values[0];

        macro_rules! fill_data {
            ($ARR_TYPE:ty) => {{
                let arr = array.as_primitive::<$ARR_TYPE>();
                for (i, val) in arr.values().iter().enumerate() {
                    let key = *val as u64;
                    // Calculate index: key - offset
                    let idx = key.wrapping_sub(offset_val) as usize;
                    if idx >= data.len() {
                        // TODO: 完善报错信息
                        return internal_err!("failed build Array idx >= data.len()");
                    }

                    if data[idx] != 0 {
                        // Duplicates are not allowed, fallback to the default hash join implementation
                        return Ok(None);
                    }
                    data[idx] = (i) as u64 + 1;
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

    pub fn process_prob_side(
        &self,
        keys_values: &[ArrayRef],
        prob_side_buffer: &mut Vec<u64>,
    ) -> Result<()> {
        assert_eq!(1, keys_values.len());
        let array = &keys_values[0];

        macro_rules! fill_buffer {
            ($ARR_TYPE:ty) => {{
                let arr = array.as_primitive::<$ARR_TYPE>();
                for (i, val) in arr.values().iter().enumerate() {
                    prob_side_buffer[i] = *val as u64;
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
            _ => internal_err!(
                "Unsupported data type for ArrayKV join: {:?}",
                array.data_type()
            )?,
        }
        Ok(())
    }

    pub fn get_matched_indices_with_limit_offset(
        &self,
        prob_side_buffer: &[u64],
        batch_size: usize,
        current_offset: JoinHashMapOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Option<JoinHashMapOffset> {
        input_indices.clear();
        match_indices.clear();

        let end = (current_offset.0 + batch_size).min(prob_side_buffer.len());

        for (prob_idx, prob_val) in
            prob_side_buffer[current_offset.0..end].iter().enumerate()
        {
            let idx_in_build_side = (prob_val.wrapping_sub(self.offset())) as usize;

            if idx_in_build_side >= self.data().len()
                || self.data()[idx_in_build_side] == 0
            {
                continue;
            }
            match_indices.push(self.data()[idx_in_build_side] - 1);
            input_indices.push((prob_idx + current_offset.0) as u32);
        }

        if end == prob_side_buffer.len() {
            None
        } else {
            Some((end, None))
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
