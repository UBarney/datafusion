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

use std::mem::size_of;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryBuilder, BinaryViewBuilder, BooleanBufferBuilder,
    LargeBinaryBuilder, LargeStringBuilder, StringBuilder, StringViewBuilder,
};
use arrow::datatypes::DataType;
use datafusion_common::{Result, internal_err};
use datafusion_expr::EmitTo;

use crate::first_last::state::{ValueState, take_need};

/// A more efficient implementation of `ValueState` for "bytes" types using a contiguous buffer.
///
/// This implementation reduces allocations by using a single `Vec<u8>` for all values and
/// provides in-place updates when the new value fits within the existing group's capacity.
/// It also includes a GC mechanism to reclaim space from "dead" bytes left behind by appends.
pub(crate) struct FlattenBytesValueState {
    /// A single, contiguous flat buffer for all raw bytes.
    vals: Vec<u8>,
    /// The starting position of each group's data in the buffer.
    offsets: Vec<usize>,
    /// The logical length of the current value for each group.
    lengths: Vec<usize>,
    /// The physical space allocated for each group (enables in-place overwrites if new_len <= capacity).
    capacities: Vec<usize>,
    /// A running counter of the sum of all current lengths (used to track fragmentation and trigger GC).
    active_bytes: usize,
    nulls: BooleanBufferBuilder,
    data_type: DataType,
}

impl FlattenBytesValueState {
    const GC_THRESHOLD: usize = 1024 * 1024;

    pub(crate) fn try_new(data_type: DataType) -> Result<Self> {
        if !matches!(
            data_type,
            DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
        ) {
            return internal_err!("FlattenBytesValueState does not support {}", data_type);
        }
        Ok(Self {
            vals: vec![],
            offsets: vec![],
            lengths: vec![],
            capacities: vec![],
            active_bytes: 0,
            nulls: BooleanBufferBuilder::new(0),
            data_type,
        })
    }

    fn should_gc(&self) -> bool {
        self.vals.len() > Self::GC_THRESHOLD && self.vals.len() > self.active_bytes * 2
    }

    fn gc(&mut self) {
        let mut new_vals = Vec::with_capacity(self.active_bytes);
        for i in 0..self.offsets.len() {
            if self.lengths[i] > 0 {
                let start = self.offsets[i];
                let end = start + self.lengths[i];
                let new_offset = new_vals.len();
                new_vals.extend_from_slice(&self.vals[start..end]);
                self.offsets[i] = new_offset;
                self.capacities[i] = self.lengths[i];
            } else {
                self.offsets[i] = 0;
                self.capacities[i] = 0;
            }
        }
        self.vals = new_vals;
    }
}

impl ValueState for FlattenBytesValueState {
    fn resize(&mut self, new_size: usize) {
        let old_size = self.offsets.len();
        if new_size < old_size {
            for i in new_size..old_size {
                self.active_bytes -= self.lengths[i];
            }
        }
        self.offsets.resize(new_size, 0);
        self.lengths.resize(new_size, 0);
        self.capacities.resize(new_size, 0);
        self.nulls.resize(new_size);
    }

    fn update(&mut self, group_idx: usize, array: &ArrayRef, idx: usize) -> Result<()> {
        if array.is_null(idx) {
            self.active_bytes -= self.lengths[group_idx];
            self.lengths[group_idx] = 0;
            self.nulls.set_bit(group_idx, false);
        } else {
            let val = match self.data_type {
                DataType::Utf8 => array.as_string::<i32>().value(idx).as_bytes(),
                DataType::LargeUtf8 => array.as_string::<i64>().value(idx).as_bytes(),
                DataType::Utf8View => array.as_string_view().value(idx).as_bytes(),
                DataType::Binary => array.as_binary::<i32>().value(idx),
                DataType::LargeBinary => array.as_binary::<i64>().value(idx),
                DataType::BinaryView => array.as_binary_view().value(idx),
                _ => unreachable!(),
            };

            let new_len = val.len();
            if new_len <= self.capacities[group_idx] {
                // In-place Overwrite
                let start = self.offsets[group_idx];
                self.vals[start..start + new_len].copy_from_slice(val);
                self.active_bytes = self.active_bytes - self.lengths[group_idx] + new_len;
                self.lengths[group_idx] = new_len;
            } else {
                // Append
                let new_offset = self.vals.len();
                self.vals.extend_from_slice(val);
                self.active_bytes = self.active_bytes - self.lengths[group_idx] + new_len;
                self.offsets[group_idx] = new_offset;
                self.lengths[group_idx] = new_len;
                self.capacities[group_idx] = new_len;
            }
            self.nulls.set_bit(group_idx, true);
        }

        if self.should_gc() {
            self.gc();
        }
        Ok(())
    }

    fn take(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let num_emit = match emit_to {
            EmitTo::All => self.offsets.len(),
            EmitTo::First(n) => n,
        };

        let nulls = take_need(&mut self.nulls, emit_to);
        let offsets = emit_to.take_needed(&mut self.offsets);
        let lengths = emit_to.take_needed(&mut self.lengths);
        let _capacities = emit_to.take_needed(&mut self.capacities);

        let mut total_len = 0;
        for &l in &lengths {
            total_len += l;
        }
        self.active_bytes -= total_len;

        let res: ArrayRef = match self.data_type {
            DataType::Utf8 => {
                let mut builder = StringBuilder::with_capacity(num_emit, total_len);
                for i in 0..num_emit {
                    if nulls.value(i) {
                        let start = offsets[i];
                        let end = start + lengths[i];
                        // SAFETY: bytes from Utf8 array
                        let s = unsafe { std::str::from_utf8_unchecked(&self.vals[start..end]) };
                        builder.append_value(s);
                    } else {
                        builder.append_null();
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::LargeUtf8 => {
                let mut builder = LargeStringBuilder::with_capacity(num_emit, total_len);
                for i in 0..num_emit {
                    if nulls.value(i) {
                        let start = offsets[i];
                        let end = start + lengths[i];
                        let s = unsafe { std::str::from_utf8_unchecked(&self.vals[start..end]) };
                        builder.append_value(s);
                    } else {
                        builder.append_null();
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::Utf8View => {
                let mut builder = StringViewBuilder::with_capacity(num_emit);
                for i in 0..num_emit {
                    if nulls.value(i) {
                        let start = offsets[i];
                        let end = start + lengths[i];
                        let s = unsafe { std::str::from_utf8_unchecked(&self.vals[start..end]) };
                        builder.append_value(s);
                    } else {
                        builder.append_null();
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::Binary => {
                let mut builder = BinaryBuilder::with_capacity(num_emit, total_len);
                for i in 0..num_emit {
                    if nulls.value(i) {
                        let start = offsets[i];
                        let end = start + lengths[i];
                        builder.append_value(&self.vals[start..end]);
                    } else {
                        builder.append_null();
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::LargeBinary => {
                let mut builder = LargeBinaryBuilder::with_capacity(num_emit, total_len);
                for i in 0..num_emit {
                    if nulls.value(i) {
                        let start = offsets[i];
                        let end = start + lengths[i];
                        builder.append_value(&self.vals[start..end]);
                    } else {
                        builder.append_null();
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::BinaryView => {
                let mut builder = BinaryViewBuilder::with_capacity(num_emit);
                for i in 0..num_emit {
                    if nulls.value(i) {
                        let start = offsets[i];
                        let end = start + lengths[i];
                        builder.append_value(&self.vals[start..end]);
                    } else {
                        builder.append_null();
                    }
                }
                Arc::new(builder.finish())
            }
            _ => {
                return internal_err!(
                    "Unsupported data type for FlattenBytesValueState: {}",
                    self.data_type
                );
            }
        };

        // If we emitted everything, we can just clear vals.
        if self.offsets.is_empty() {
            self.vals.clear();
        }

        Ok(res)
    }

    fn size(&self) -> usize {
        self.vals.capacity()
            + self.offsets.capacity() * size_of::<usize>()
            + self.lengths.capacity() * size_of::<usize>()
            + self.capacities.capacity() * size_of::<usize>()
            + self.nulls.capacity() / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;

    #[test]
    fn test_utf8_basics() -> Result<()> {
        let mut state = FlattenBytesValueState::try_new(DataType::Utf8)?;
        state.resize(2);

        let array: ArrayRef = Arc::new(StringArray::from(vec![Some("hello"), Some("world")]));
        state.update(0, &array, 0)?;
        state.update(1, &array, 1)?;

        let result = state.take(EmitTo::All)?;
        let result = result.as_string::<i32>();
        assert_eq!(result.value(0), "hello");
        assert_eq!(result.value(1), "world");
        // After emitting all, vals should be cleared
        assert_eq!(state.vals.len(), 0);
        assert_eq!(state.active_bytes, 0);

        Ok(())
    }

    #[test]
    fn test_utf8_update_logic() -> Result<()> {
        let mut state = FlattenBytesValueState::try_new(DataType::Utf8)?;
        state.resize(1);

        let array: ArrayRef = Arc::new(StringArray::from(vec![
            Some("short"),         // 5 bytes
            Some("longer_string"),  // 13 bytes
            Some("medium"),         // 6 bytes
            Some("s"),              // 1 byte
        ]));

        // 1. Initial append
        state.update(0, &array, 0)?;
        assert_eq!(state.lengths[0], 5);
        assert_eq!(state.capacities[0], 5);
        assert_eq!(state.active_bytes, 5);
        let offset0 = state.offsets[0];

        // 2. Append (longer > short)
        state.update(0, &array, 1)?;
        assert_eq!(state.lengths[0], 13);
        assert_eq!(state.capacities[0], 13);
        assert_eq!(state.active_bytes, 13);
        assert_ne!(state.offsets[0], offset0); // Should have moved
        let offset1 = state.offsets[0];

        // 3. In-place overwrite (medium < longer_string)
        state.update(0, &array, 2)?;
        assert_eq!(state.lengths[0], 6);
        assert_eq!(state.capacities[0], 13); // Capacity remains 13
        assert_eq!(state.active_bytes, 6);
        assert_eq!(state.offsets[0], offset1); // Should NOT have moved

        // 4. In-place overwrite (s < medium)
        state.update(0, &array, 3)?;
        assert_eq!(state.lengths[0], 1);
        assert_eq!(state.capacities[0], 13);
        assert_eq!(state.active_bytes, 1);
        assert_eq!(state.offsets[0], offset1);

        let result = state.take(EmitTo::All)?;
        assert_eq!(result.as_string::<i32>().value(0), "s");
        Ok(())
    }

    #[test]
    fn test_utf8_nulls() -> Result<()> {
        let mut state = FlattenBytesValueState::try_new(DataType::Utf8)?;
        state.resize(2);

        let array: ArrayRef = Arc::new(StringArray::from(vec![Some("val"), None]));

        state.update(0, &array, 0)?;
        assert_eq!(state.active_bytes, 3);
        assert!(state.nulls.as_slice()[0] & 1 != 0);

        state.update(0, &array, 1)?; // Set to null
        assert_eq!(state.active_bytes, 0);
        assert_eq!(state.lengths[0], 0);
        assert!(state.nulls.as_slice()[0] & 1 == 0);

        let result = state.take(EmitTo::All)?;
        assert!(result.is_null(0));
        Ok(())
    }

    #[test]
    fn test_utf8_gc() -> Result<()> {
        let mut state = FlattenBytesValueState::try_new(DataType::Utf8)?;
        state.resize(2);

        // We need to exceed 1MB and have > 50% garbage.
        let large_val = "a".repeat(600 * 1024); // 600KB
        let array: ArrayRef = Arc::new(StringArray::from(vec![Some(large_val.as_str()), Some("small")]));

        // 1. Fill state
        state.update(0, &array, 0)?; // Group 0: 600KB
        state.update(1, &array, 1)?; // Group 1: 5 bytes
        assert_eq!(state.active_bytes, 600 * 1024 + 5);

        // 2. Create more garbage
        let very_large_val = "b".repeat(1100 * 1024); // 1.1MB
        let array2: ArrayRef = Arc::new(StringArray::from(vec![Some(very_large_val.as_str())]));
        state.update(0, &array2, 0)?; 
        // vals now contains: [old 600KB (garbage)] [5B (valid)] [1.1MB (valid)]
        // vals.len() ≈ 1.7MB, active_bytes ≈ 1.1MB. Not yet triggering GC (1.7 < 1.1 * 2)

        // 3. Overwrite with small value to trigger GC
        state.update(0, &array, 1)?; 
        // Group 0 in-place overwrites 1.1MB space.
        // vals.len() ≈ 1.7MB, active_bytes = 5 + 5 = 10 bytes.
        // 1.7MB > 1MB AND 1.7MB > 10 * 2. GC triggered!

        assert!(state.vals.len() < 100); // GC should have shrunk it significantly
        assert_eq!(state.active_bytes, 10);
        
        let result = state.take(EmitTo::All)?;
        let result = result.as_string::<i32>();
        assert_eq!(result.value(0), "small");
        assert_eq!(result.value(1), "small");
        
        Ok(())
    }

    #[test]
    fn test_utf8_emit_first() -> Result<()> {
        let mut state = FlattenBytesValueState::try_new(DataType::Utf8)?;
        state.resize(3);

        let array: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        state.update(0, &array, 0)?;
        state.update(1, &array, 1)?;
        state.update(2, &array, 2)?;
        assert_eq!(state.active_bytes, 3);

        // Take first 2
        let result = state.take(EmitTo::First(2))?;
        let result = result.as_string::<i32>();
        assert_eq!(result.len(), 2);
        assert_eq!(result.value(0), "a");
        assert_eq!(result.value(1), "b");
        
        assert_eq!(state.active_bytes, 1);
        assert_eq!(state.offsets.len(), 1);
        assert_eq!(state.lengths[0], 1);

        // Take remaining
        let result = state.take(EmitTo::All)?;
        assert_eq!(result.as_string::<i32>().value(0), "c");
        assert_eq!(state.active_bytes, 0);

        Ok(())
    }
}
