# REPL Internals

## DataFusion Table Registration

### TableProviderFactory

DataFusion uses the [`TableProviderFactory`](https://docs.rs/datafusion/latest/datafusion/catalog/trait.TableProviderFactory.html) trait to handle `CREATE EXTERNAL TABLE` statements. When a user executes:

```sql
CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION 'data/weather.zarr';
```

DataFusion:
1. Parses the SQL into a [`CreateExternalTable`](https://docs.rs/datafusion/latest/datafusion/logical_expr/struct.CreateExternalTable.html) struct
2. Looks up the factory registered for `"ZARR"` in the session's table factories map
3. Calls `factory.create(state, cmd)` to obtain an `Arc<dyn TableProvider>`
4. Registers the provider in the catalog under the given table name

Our `ZarrTableFactory` implements this trait:

```rust
impl TableProviderFactory for ZarrTableFactory {
    async fn create(&self, _state: &dyn Session, cmd: &CreateExternalTable)
        -> Result<Arc<dyn TableProvider>>
    {
        let schema = Arc::new(infer_schema(&cmd.location)?);
        Ok(Arc::new(ZarrTable::new(schema, &cmd.location)))
    }
}
```

### Factory Registration

Factories are registered via [`SessionStateBuilder::with_table_factories`](https://docs.rs/datafusion/latest/datafusion/execution/session_state/struct.SessionStateBuilder.html):

```rust
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_table_factories(HashMap::from([
        ("ZARR".to_string(), Arc::new(ZarrTableFactory) as _),
    ]))
    .build();
```

The key `"ZARR"` maps to the `STORED AS ZARR` clause in SQL.

## Catalog System

### Hierarchy

DataFusion organizes tables in a three-level hierarchy:

```
CatalogProvider (default: "datafusion")
└── SchemaProvider (default: "public")
    └── TableProvider (e.g., "weather")
```

When `CREATE EXTERNAL TABLE weather ...` executes, the table is registered at:
- Catalog: `datafusion`
- Schema: `public`
- Table: `weather`

### Information Schema

The [`information_schema`](https://datafusion.apache.org/user-guide/sql/information_schema.html) is a virtual schema containing metadata tables:

- `information_schema.tables` — all registered tables
- `information_schema.columns` — column metadata
- `information_schema.views` — registered views

`SHOW TABLES` is syntactic sugar that queries `information_schema.tables`. It requires:

```rust
SessionConfig::new().with_information_schema(true)
```

Without this, DataFusion returns:
```
Error: SHOW TABLES is not supported unless information_schema is enabled
```

### Table Lifecycle

| SQL | Catalog Operation |
|-----|-------------------|
| `CREATE EXTERNAL TABLE t ...` | `schema.register_table("t", provider)` |
| `DROP TABLE t` | `schema.deregister_table("t")` |
| `SHOW TABLES` | Query `information_schema.tables` |

## Session State

[`SessionState`](https://docs.rs/datafusion/latest/datafusion/execution/session_state/struct.SessionState.html) holds all runtime configuration:

- **Catalog list** — registered catalogs and schemas
- **Table factories** — `STORED AS <format>` handlers
- **Config** — execution settings, information_schema flag
- **Runtime environment** — memory pools, object stores

The CLI builds state with:

```rust
let config = SessionConfig::new().with_information_schema(true);
let state = SessionStateBuilder::new()
    .with_default_features()      // registers default catalogs, optimizers
    .with_config(config)          // enables information_schema
    .with_table_factories(...)    // registers ZARR factory
    .build();

let ctx = SessionContext::new_with_state(state);
```

## References

- [TableProviderFactory trait](https://docs.rs/datafusion/latest/datafusion/catalog/trait.TableProviderFactory.html)
- [CreateExternalTable struct](https://docs.rs/datafusion/latest/datafusion/logical_expr/struct.CreateExternalTable.html)
- [SessionStateBuilder](https://docs.rs/datafusion/latest/datafusion/execution/session_state/struct.SessionStateBuilder.html)
- [Information Schema](https://datafusion.apache.org/user-guide/sql/information_schema.html)
- [Catalog and Schema](https://datafusion.apache.org/library-user-guide/catalogs.html)
