mod common;

use datafusion::logical_expr::Expr;

const LOCAL_STORE: &str = "data/era5_sst_local.zarr";

// Unbounded: 63,864 Dec hours × 721 × 1440 = 66B rows — intentionally not executed
const SQL_UNBOUNDED: &str = "SELECT COUNT(*) FROM era5 WHERE EXTRACT(MONTH FROM time) = 12";

// Bounded: same month filter + Niño 3.4 spatial box — safe to execute
const SQL_BOUNDED: &str = "
    SELECT time
    FROM era5
    WHERE EXTRACT(MONTH FROM time) = 12
      AND latitude  BETWEEN -5.0 AND  5.0
      AND longitude BETWEEN 190.0 AND 240.0
    LIMIT 5
";

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    common::init_tracing();
    let ctx = common::create_remote_context();

    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION '{LOCAL_STORE}'"
    ))
    .await?
    .collect()
    .await?;

    // -------------------------------------------------------------------------
    // Step 1: unoptimized vs optimized logical plan
    // -------------------------------------------------------------------------
    println!("=== Unoptimized logical plan ===");
    let df = ctx.sql(SQL_UNBOUNDED).await?;
    println!("{}\n", df.logical_plan());

    println!("=== Optimized logical plan ===");
    let df = ctx.sql(SQL_UNBOUNDED).await?;
    let optimized = df.into_optimized_plan()?;
    println!("{optimized}\n");

    println!("=== Filters pushed to TableScan (optimized) ===");
    inspect_filters(&optimized);
    println!();

    // -------------------------------------------------------------------------
    // Step 2: safe bounded execution — confirms the filter works end-to-end
    // Note: SQL_UNBOUNDED is not executed — 66B rows without spatial bounds
    //       would OOM on any machine.
    // -------------------------------------------------------------------------
    println!("=== Bounded execution (LIMIT 5, Niño 3.4 box) ===");
    println!("SQL:{SQL_BOUNDED}");
    match ctx.sql(SQL_BOUNDED).await?.collect().await {
        Ok(batches) => {
            for batch in &batches {
                arrow::util::pretty::print_batches(&[batch.clone()]).ok();
            }
        }
        Err(e) => println!("Execution error: {e}"),
    }

    Ok(())
}

fn inspect_filters(plan: &datafusion::logical_expr::LogicalPlan) {
    use datafusion::logical_expr::LogicalPlan;
    match plan {
        LogicalPlan::TableScan(scan) => {
            if scan.filters.is_empty() {
                println!("  (none)");
            }
            for (i, f) in scan.filters.iter().enumerate() {
                println!("  [{i}] {f}   ({:?})", expr_type_name(f));
            }
        }
        other => {
            for input in other.inputs() {
                inspect_filters(input);
            }
        }
    }
}

fn expr_type_name(expr: &Expr) -> &'static str {
    match expr {
        Expr::BinaryExpr(_) => "BinaryExpr",
        Expr::Column(_) => "Column",
        Expr::Literal(_, _) => "Literal",
        Expr::ScalarFunction(_) => "ScalarFunction",
        Expr::Between(_) => "Between",
        Expr::Cast(_) => "Cast",
        _ => "Other",
    }
}
