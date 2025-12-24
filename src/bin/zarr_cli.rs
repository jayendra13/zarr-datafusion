use std::path::Path;
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

struct TableInfo {
    name: &'static str,
    path: &'static str,
}

const TABLES: &[TableInfo] = &[
    TableInfo { name: "synthetic", path: "data/synthetic.zarr" },
    TableInfo { name: "era5", path: "data/era5.zarr" },
];

fn register_table(
    ctx: &SessionContext,
    info: &TableInfo,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if !Path::new(info.path).exists() {
        return Ok(()); // Skip if data doesn't exist
    }

    let schema = Arc::new(infer_schema(info.path)?);
    let table = Arc::new(ZarrTable::new(schema.clone(), info.path));
    ctx.register_table(info.name, table)?;

    println!(
        "  {} ({}) - columns: {}",
        info.name,
        info.path,
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ctx = SessionContext::new();

    println!("Zarr-DataFusion CLI");
    println!("Registered tables:");

    let mut registered = Vec::new();
    for info in TABLES {
        if Path::new(info.path).exists() {
            register_table(&ctx, info)?;
            registered.push(info.name);
        }
    }

    if registered.is_empty() {
        println!("  (none) - run ./scripts/generate_data.sh first");
    }

    println!("\nType SQL queries or 'help' for commands.\n");

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
                    for name in &registered {
                        println!("  {}", name);
                    }
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

Example queries (synthetic data):
  SELECT * FROM synthetic LIMIT 10;
  SELECT AVG(temperature) FROM synthetic GROUP BY lat, lon;

Example queries (ERA5 data):
  SELECT * FROM era5 LIMIT 10;
  SELECT hybrid, AVG(temperature) as avg_temp FROM era5 GROUP BY hybrid;
  SELECT latitude, longitude, temperature FROM era5 WHERE temperature > 300 LIMIT 10;
"#
    );
}
