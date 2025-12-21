use std::sync::Arc;

use datafusion::prelude::SessionContext;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::zarr_reader::zarr_weather_schema;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = SessionContext::new();

    // Register the weather table by default
    let schema = Arc::new(zarr_weather_schema());
    let table = Arc::new(ZarrTable::new(schema, "data/weather.zarr"));
    ctx.register_table("weather", table)?;

    println!("Zarr-DataFusion CLI");
    println!("Registered table: weather (from data/weather.zarr)");
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
  SELECT timestamp, AVG(temperature) FROM weather GROUP BY timestamp;
  SELECT * FROM weather WHERE temperature > 20;
"#
    );
}
