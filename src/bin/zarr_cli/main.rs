mod highlight;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use highlight::SqlHelper;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use zarr_datafusion::datasource::factory::ZarrTableFactory;

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

                // Execute SQL
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
                            }
                        } else if let Err(e) = df.show().await {
                            eprintln!("Error displaying results: {e}");
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
