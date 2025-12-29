mod highlight;

use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::util::pretty::print_batches;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::{SessionConfig, SessionContext};
use highlight::SqlHelper;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use zarr_datafusion::datasource::factory::ZarrTableFactory;
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;
use zarr_datafusion::reader::stats::{format_bytes, SharedIoStats};

const HISTORY_FILE: &str = ".zarr_cli_history";

/// Get the path to the history file (~/.zarr_cli_history)
fn get_history_path() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(HISTORY_FILE)
}

// Why `Send + Sync` in the error type?
//
// Even though we don't explicitly spawn threads, errors must be `Send + Sync` because:
// 1. Tokio runtime is multi-threaded by default - tasks may move between threads at .await points
// 2. DataFusion uses parallelism internally for query execution
// 3. Any error held across an .await must be Send to satisfy the async runtime
//
// References:
// - Tokio multi-threaded runtime: https://tokio.rs/tokio/tutorial/spawning#concurrency
// - Rust Send/Sync traits: https://doc.rust-lang.org/nomicon/send-and-sync.html
// - DataFusion async execution: https://docs.rs/datafusion/latest/datafusion/execution/context/struct.SessionContext.html

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = SessionConfig::new().with_information_schema(true);
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_table_factories(HashMap::from([(
            "ZARR".to_string(),
            Arc::new(ZarrTableFactory) as _,
        )]))
        .build();
    let ctx = SessionContext::new_with_state(state);

    println!("Zarr-DataFusion CLI");
    println!("\nType SQL queries or 'help' for commands.\n");

    let helper = SqlHelper::new();
    let mut rl = Editor::new()?;
    rl.set_helper(Some(helper));

    // Load command history (ignore error if file doesn't exist)
    let history_path = get_history_path();
    let _ = rl.load_history(&history_path);

    loop {
        match rl.readline("zarr> ") {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(line);

                if line.eq_ignore_ascii_case("quit") || line.eq_ignore_ascii_case("exit") {
                    break;
                }

                if line.eq_ignore_ascii_case("help") {
                    print_help();
                    continue;
                }

                if line.starts_with("\\d") || line.eq_ignore_ascii_case("show tables") {
                    match ctx.sql("SHOW TABLES").await {
                        Ok(df) => {
                            if let Err(e) = df.show().await {
                                eprintln!("Error: {e}");
                            }
                        }
                        Err(e) => eprintln!("Error: {e}"),
                    }
                    continue;
                }

                // Execute SQL with timing
                let start = Instant::now();
                match ctx.sql(line).await {
                    Ok(df) => {
                        // DDL statements return empty results - don't show the empty table
                        let line_upper = line.to_uppercase();
                        let is_ddl = line_upper.starts_with("CREATE ")
                            || line_upper.starts_with("DROP ")
                            || line_upper.starts_with("ALTER ");

                        if is_ddl {
                            // Execute DDL silently - only show errors
                            if let Err(e) = df.collect().await {
                                eprintln!("Error: {e}");
                            } else {
                                let elapsed = start.elapsed();
                                println!("OK ({:.3}s)", elapsed.as_secs_f64());
                            }
                        } else {
                            // Create physical plan to access ZarrExec for I/O stats
                            match df.create_physical_plan().await {
                                Ok(plan) => {
                                    // Find ZarrExec in the plan tree to get I/O stats
                                    let io_stats = find_zarr_exec_stats(&plan);

                                    // Start live stats display if we have ZarrExec stats
                                    // Only show live updates when stdout is a terminal
                                    let stop_flag = Arc::new(AtomicBool::new(false));
                                    let is_tty = io::stdout().is_terminal();
                                    let live_task = if is_tty {
                                        io_stats.as_ref().map(|stats| {
                                            spawn_live_stats(stats.clone(), stop_flag.clone())
                                        })
                                    } else {
                                        None
                                    };

                                    // Execute using the same plan (so stats are populated)
                                    let task_ctx = ctx.task_ctx();
                                    let result = collect(plan, task_ctx).await;

                                    // Stop live stats display
                                    stop_flag.store(true, Ordering::Relaxed);
                                    if let Some(task) = live_task {
                                        let _ = task.await;
                                    }

                                    match result {
                                        Ok(batches) => {
                                            let elapsed = start.elapsed();
                                            let row_count: usize =
                                                batches.iter().map(|b| b.num_rows()).sum();

                                            // Clear the live stats line if we were showing it
                                            if is_tty && io_stats.is_some() {
                                                print!("\r\x1b[K");
                                                let _ = io::stdout().flush();
                                            }

                                            // Print results table
                                            if let Err(e) = print_batches(&batches) {
                                                eprintln!("Error displaying results: {e}");
                                            } else {
                                                // Print compact stats line
                                                print_stats_line(
                                                    row_count,
                                                    elapsed.as_secs_f64(),
                                                    io_stats.as_ref(),
                                                );
                                            }
                                        }
                                        Err(e) => eprintln!("Error executing query: {e}"),
                                    }
                                }
                                Err(e) => eprintln!("Error creating plan: {e}"),
                            }
                        }
                    }
                    Err(e) => eprintln!("SQL Error: {e}"),
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Error: {err}");
                break;
            }
        }
    }

    // Save command history
    if let Err(e) = rl.save_history(&history_path) {
        eprintln!("Warning: Could not save history: {e}");
    }

    println!("Goodbye!");
    Ok(())
}

fn print_help() {
    println!(
        r#"
  Zarr-DataFusion CLI Commands:
    <SQL>           Execute a SQL query
    show tables     List registered tables
    \d              List registered tables
    help            Show this help
    quit/exit       Exit the CLI

  Loading data:
    CREATE EXTERNAL TABLE <name> STORED AS ZARR LOCATION '<path>';
    DROP TABLE <name>;

  Example:
    CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION 'data/synthetic.zarr';
    SELECT * FROM weather LIMIT 10;
    SELECT AVG(temperature) FROM weather GROUP BY lat, lon;
    DROP TABLE weather;
  "#
    );
}

/// Recursively search the execution plan tree for ZarrExec and return its I/O stats
fn find_zarr_exec_stats(plan: &Arc<dyn ExecutionPlan>) -> Option<SharedIoStats> {
    // Check if this node is ZarrExec
    if let Some(zarr_exec) = plan.as_any().downcast_ref::<ZarrExec>() {
        return Some(zarr_exec.io_stats());
    }

    // Recursively check children
    for child in plan.children() {
        if let Some(stats) = find_zarr_exec_stats(child) {
            return Some(stats);
        }
    }

    None
}

/// Print compact stats line: "5 rows · 3 arrays · 6.70 KB disk · 13.92 KB mem · 0.013s"
fn print_stats_line(row_count: usize, elapsed_secs: f64, io_stats: Option<&SharedIoStats>) {
    let mut parts = vec![format!(
        "{} row{}",
        row_count,
        if row_count == 1 { "" } else { "s" }
    )];

    if let Some(stats) = io_stats {
        let total_arrays =
            stats.coord_arrays.load(Ordering::Relaxed) + stats.data_arrays.load(Ordering::Relaxed);
        let disk_bytes = stats.total_disk_bytes();
        let mem_bytes = stats.total_bytes();

        parts.push(format!(
            "{} array{}",
            total_arrays,
            if total_arrays == 1 { "" } else { "s" }
        ));
        parts.push(format!("{} disk", format_bytes(disk_bytes)));
        parts.push(format!("{} mem", format_bytes(mem_bytes)));
    }

    parts.push(format!("{:.3}s", elapsed_secs));

    println!("\n{}", parts.join(" · "));
}

/// Spawn a background task that displays live I/O stats
fn spawn_live_stats(stats: SharedIoStats, stop: Arc<AtomicBool>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while !stop.load(Ordering::Relaxed) {
            let arrays = stats.coord_arrays.load(Ordering::Relaxed)
                + stats.data_arrays.load(Ordering::Relaxed);
            let disk_bytes = stats.total_disk_bytes();

            // Use \r to overwrite line, \x1b[K to clear to end of line
            print!(
                "\r{} array{} · {} disk...\x1b[K",
                arrays,
                if arrays == 1 { "" } else { "s" },
                format_bytes(disk_bytes)
            );
            let _ = io::stdout().flush();

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
}
