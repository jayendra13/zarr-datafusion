use std::sync::Arc;

use datafusion::prelude::SessionContext;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema;

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
    let ctx = SessionContext::new();

    let store_path = "data/weather.zarr";
    let schema = Arc::new(infer_schema(store_path)?);
    let table = Arc::new(ZarrTable::new(schema.clone(), store_path));
    ctx.register_table("weather", table)?;

    println!("Zarr-DataFusion CLI");
    println!("Registered table: weather (from {})", store_path);
    println!(
        "Columns: {}",
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Type SQL queries or 'quit' to exit.\n");

    let mut rl = DefaultEditor::new()?;

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
                    println!("Tables:");
                    println!("  weather");
                    continue;
                }

                // Execute SQL
                match ctx.sql(line).await {
                    Ok(df) => {
                        if let Err(e) = df.show().await {
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

    println!("Goodbye!");
    Ok(())
}

fn print_help() {
    println!(
        r#"
Zarr-DataFusion CLI Commands:
  <SQL>         Execute a SQL query
  show tables   List registered tables
  \d            List registered tables
  help          Show this help
  quit/exit     Exit the CLI

Example queries:
  SELECT * FROM weather LIMIT 10;
  SELECT AVG(temperature) FROM weather GROUP BY lat, lon;
"#
    );
}
