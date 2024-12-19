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

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_catalog::TableFunctionImpl;
use datafusion_catalog::TableProvider;
use datafusion_common::{not_impl_err, plan_err, Result, ScalarValue};
use datafusion_expr::{Expr, TableType};
use datafusion_physical_plan::memory::{LazyBatchGenerator, LazyMemoryExec};
use datafusion_physical_plan::ExecutionPlan;
use parking_lot::RwLock;
use std::fmt;
use std::sync::Arc;

/// Table that generates a series of integers from `start`(inclusive) to `end`(inclusive)
#[derive(Debug, Clone)]
struct GenerateSeriesTable {
    schema: SchemaRef,
    start: i64,
    end: i64,
    step: i64,
}

/// Table state that generates a series of integers from `start`(inclusive) to `end`(inclusive)
#[derive(Debug, Clone)]
struct GenerateSeriesState {
    schema: SchemaRef,
    start: i64, // Kept for display
    end: i64,
    step: i64,
    batch_size: usize,

    /// Tracks current position when generating table
    next: i64,
}

impl GenerateSeriesState {
    fn reach_end(&self, val: i64) -> bool {
        if self.step > 0 {
            return val > self.end;
        }

        return val < self.end;
    }
}

/// Detail to display for 'Explain' plan
impl fmt::Display for GenerateSeriesState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "generate_series: start={}, end={}, batch_size={}",
            self.start, self.end, self.batch_size
        )
    }
}

impl LazyBatchGenerator for GenerateSeriesState {
    fn generate_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let mut buf = Vec::with_capacity(self.batch_size);

        // TODO: rewrite with high performance
        // remove if cond
        while buf.len() < self.batch_size && !self.reach_end(self.next) {
            buf.push(self.next);
            self.next += self.step;
        }
        let array = Int64Array::from(buf);

        if array.len() == 0 {
            return Ok(None);
        }

        let batch = RecordBatch::try_new(self.schema.clone(), vec![Arc::new(array)])?;

        Ok(Some(batch))
    }
}

#[async_trait]
impl TableProvider for GenerateSeriesTable {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let batch_size = state.config_options().execution.batch_size;
        Ok(Arc::new(LazyMemoryExec::try_new(
            self.schema.clone(),
            vec![Arc::new(RwLock::new(GenerateSeriesState {
                schema: self.schema.clone(),
                start: self.start,
                end: self.end,
                step: self.step,
                next: self.start,
                batch_size,
            }))],
        )?))
    }
}

#[derive(Debug)]
pub struct GenerateSeriesFunc {}

impl TableFunctionImpl for GenerateSeriesFunc {
    // Check input `exprs` type and number. Input validity check (e.g. start <= end)
    // will be performed in `TableProvider::scan`
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        // TODO: support 1 or 3 arguments following DuckDB:
        // <https://duckdb.org/docs/sql/functions/list#generate_series>

        if exprs.len() == 0 || exprs.len() > 3 {
            return plan_err!("generate_series function requires 1 to 3 arguments");
        }

        let mut normalize_args = Vec::new();
        for ele in exprs {
            match ele {
                Expr::Literal(ScalarValue::Null) => {}
                Expr::Literal(ScalarValue::Int64(Some(n))) => normalize_args.push(*n),
                _ => return plan_err!("First argument must be an integer literal"),
            };
        }

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

        if normalize_args.len() != exprs.len() {
            // containe null
            return Ok(Arc::new(GenerateSeriesTable {
                schema,
                start: 1,
                end: 0,
                step: 1,
            }));
        }

        let (start, stop, step) = match normalize_args.len() {
            1 => (0, normalize_args[0], 1),
            2 => (normalize_args[0], normalize_args[1], 1),
            3 => (normalize_args[0], normalize_args[1], normalize_args[2]),
            _ => {
                return plan_err!("generate_series function requires 1 to 3 arguments");
            }
        };

        if start > stop && step > 0 {
            return plan_err!("start is bigger than end, but increment is positive: cannot generate infinite series");
        }

        if start < stop && step < 0 {
            return plan_err!("start is smaller than end, but increment is negative: cannot generate infinite series");
        }

        if step == 0 {
            return plan_err!("step cannot be zero");
        }

        Ok(Arc::new(GenerateSeriesTable {
            schema,
            start,
            end: stop,
            step,
        }))
    }
}
