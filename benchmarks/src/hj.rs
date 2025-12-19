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

use crate::util::{BenchmarkRun, CommonOpt, QueryResult};
use datafusion::physical_plan::execute_stream;
use datafusion::{error::Result, prelude::SessionContext};
use datafusion_common::instant::Instant;
use datafusion_common::{DataFusionError, exec_datafusion_err, exec_err};
use std::path::PathBuf;
use structopt::StructOpt;

use futures::StreamExt;

// TODO: Add existence joins

/// Run the Hash Join benchmark
///
/// This micro-benchmark focuses on the performance characteristics of Hash Joins.
/// It uses simple equality predicates to ensure a hash join is selected.
/// Where we vary selectivity, we do so with additional cheap predicates that
/// do not change the join key (so the physical operator remains HashJoin).
#[derive(Debug, StructOpt, Clone)]
#[structopt(verbatim_doc_comment)]
pub struct RunOpt {
    /// Query number (between 1 and 6). If not specified, runs all queries
    #[structopt(short, long)]
    query: Option<usize>,

    /// Common options (iterations, batch size, target_partitions, etc.)
    #[structopt(flatten)]
    common: CommonOpt,

    /// Path to TPC-H data (required for TPC-H queries)
    #[structopt(parse(from_os_str), short = "p", long = "path")]
    path: Option<PathBuf>,

    /// If present, write results json here
    #[structopt(parse(from_os_str), short = "o", long = "output")]
    output_path: Option<PathBuf>,
}

/// Inline SQL queries for Hash Join benchmarks
const HASH_QUERIES: &[&str] = &[
    // Q1: Very Small Build Side (Dense)
    // Build Side: nation (25 rows) | Probe Side: customer (1.5M rows)
    r#"SELECT n_nationkey FROM nation JOIN customer ON c_nationkey = n_nationkey"#,

    // Q2: 100% Density, 100% Hit rate
    // Build Side: supplier (100k rows) | Probe Side: lineitem (60M rows)
    r#"SELECT s_suppkey FROM supplier JOIN lineitem ON s_suppkey = l_suppkey"#,

    // Q3: 100% Density, 10% Hit rate
    // Build Side: supplier (100k rows) | Probe Side: lineitem (60M rows)
    r#"SELECT l_suppkey 
    FROM lineitem 
    JOIN supplier ON s_suppkey = l_suppkey + (l_suppkey % 10) * 1000000"#,

    // Q4: 90% Density, ~100% Hit rate
    // Build Side: supplier (100k unique rows) | Probe Side: lineitem (60M rows)
    r#"SELECT l.k
    FROM (
      SELECT l_suppkey * 10 / 9 as k
      FROM lineitem
    ) l
    JOIN (
      SELECT s_suppkey * 10 / 9 as k FROM supplier
    ) s ON l.k = s.k"#,

    // Q5: 90% Density, 10% Hit rate
    // Build Side: supplier (100k unique rows, range 111k) | Probe Side: lineitem (60M rows)
    r#"SELECT l.k
    FROM (
      SELECT CASE WHEN l_suppkey % 10 = 0 
                  THEN l_suppkey * 10 / 9
                  ELSE (l_suppkey / 10) * 10 + 9
             END as k
      FROM lineitem
    ) l
    JOIN (
      SELECT s_suppkey * 10 / 9 as k FROM supplier
    ) s ON l.k = s.k"#,

    // Q6: Extremely Sparse Build Side (Small but Wide)
    // Build Side: 512 rows, Range: 0 to 51,200,000 | Probe Side: 10M rows
    r#"SELECT build.value
    FROM range(0, 512 * 100000, 100000) AS build
    JOIN range(10000000) AS probe ON build.value = probe.value"#,
];

impl RunOpt {
    pub async fn run(self) -> Result<()> {
        println!("Running Hash Join benchmarks with the following options: {self:#?}\n");

        let query_range = match self.query {
            Some(query_id) => {
                if query_id >= 1 && query_id <= HASH_QUERIES.len() {
                    query_id..=query_id
                } else {
                    return exec_err!(
                        "Query {query_id} not found. Available queries: 1 to {}",
                        HASH_QUERIES.len()
                    );
                }
            }
            None => 1..=HASH_QUERIES.len(),
        };

        let config = self.common.config()?;
        let rt_builder = self.common.runtime_env_builder()?;
        let ctx = SessionContext::new_with_config_rt(config, rt_builder.build_arc()?);

        if let Some(path) = &self.path {
            for table in &["lineitem", "supplier", "nation", "customer"] {
                let table_path = path.join(table);
                if !table_path.exists() {
                    return exec_err!("TPC-H table {} not found at {:?}", table, table_path);
                }
                ctx.register_parquet(*table, table_path.to_str().unwrap(), Default::default())
                    .await?;
            }
        }

        let mut benchmark_run = BenchmarkRun::new();

        for query_id in query_range {
            let query_index = query_id - 1;
            let sql = HASH_QUERIES[query_index];

            benchmark_run.start_new_case(&format!("Query {query_id}"));
            let query_run = self.benchmark_query(sql, &query_id.to_string(), &ctx).await;
            match query_run {
                Ok(query_results) => {
                    for iter in query_results {
                        benchmark_run.write_iter(iter.elapsed, iter.row_count);
                    }
                }
                Err(e) => {
                    return Err(DataFusionError::Context(
                        format!("Hash Join benchmark Q{query_id} failed with error:"),
                        Box::new(e),
                    ));
                }
            }
        }

        benchmark_run.maybe_write_json(self.output_path.as_ref())?;
        Ok(())
    }

    /// Validates that the physical plan uses a HashJoin, then executes.
    async fn benchmark_query(
        &self,
        sql: &str,
        query_name: &str,
        ctx: &SessionContext,
    ) -> Result<Vec<QueryResult>> {
        let mut query_results = vec![];

        // Build/validate plan
        let df = ctx.sql(sql).await?;
        let physical_plan = df.create_physical_plan().await?;
        let plan_string = format!("{physical_plan:#?}");

        if !plan_string.contains("HashJoinExec") {
            return Err(exec_datafusion_err!(
                "Query {query_name} does not use Hash Join. Physical plan: {plan_string}"
            ));
        }

        // Execute without buffering
        for i in 0..self.common.iterations {
            let start = Instant::now();
            let row_count = Self::execute_sql_without_result_buffering(sql, ctx).await?;
            let elapsed = start.elapsed();

            println!(
                "Query {query_name} iteration {i} returned {row_count} rows in {elapsed:?}"
            );

            query_results.push(QueryResult { elapsed, row_count });
        }

        Ok(query_results)
    }

    /// Executes the SQL query and drops each batch to avoid result buffering.
    async fn execute_sql_without_result_buffering(
        sql: &str,
        ctx: &SessionContext,
    ) -> Result<usize> {
        let mut row_count = 0;

        let df = ctx.sql(sql).await?;
        let physical_plan = df.create_physical_plan().await?;
        let mut stream = execute_stream(physical_plan, ctx.task_ctx())?;

        while let Some(batch) = stream.next().await {
            row_count += batch?.num_rows();
            // Drop batches immediately to minimize memory pressure
        }

        Ok(row_count)
    }
}
