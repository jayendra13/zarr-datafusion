//! Language definition for e-graph query optimization
//!
//! This module defines `ZarrPlan`, a language for representing DataFusion
//! logical plans in egg's e-graph format.
//!
//! # Supported Operators
//!
//! - **Relational**: scan, filter, project, aggregate, join, sort, limit, union
//! - **Expressions**: arithmetic (+, -, *, /, %), comparison (=, <>, <, <=, >, >=)
//! - **Logical**: and, or, not, is_null, is_not_null
//! - **Aggregates**: count, sum, avg, min, max
//! - **Zarr-specific**: zarr_scan, coord_filter, resample

use egg::{define_language, Id, Symbol};

define_language! {
    pub enum ZarrPlan {
        // Relational Operators
        "scan" = Scan([Id; 2]),            // [table_ref, projection]
        "filter" = Filter([Id; 2]),        // [input, predicate]
        "project" = Project([Id; 2]),      // [input, expressions]
        "aggregate" = Aggregate([Id; 3]),  // [input, group_by, aggregates]
        "sort" = Sort([Id; 2]),            // [input, sort_keys]
        "limit" = Limit([Id; 2]),          // [input, count]
        "distinct" = Distinct([Id; 1]),    // [input]
        "union" = Union([Id; 2]),          // [left, right]

        // Join Operators
        "inner_join" = InnerJoin([Id; 3]), // [left, right, condition]
        "left_join" = LeftJoin([Id; 3]),   // [left, right, condition]
        "right_join" = RightJoin([Id; 3]), // [left, right, condition]
        "full_join" = FullJoin([Id; 3]),   // [left, right, condition]
        "cross_join" = CrossJoin([Id; 2]), // [left, right]
        "semi_join" = SemiJoin([Id; 3]),   // [left, right, condition]
        "anti_join" = AntiJoin([Id; 3]),   // [left, right, condition]

        // Arithmetic Expressions
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Mod([Id; 2]),
        "neg" = Neg([Id; 1]),

        // Comparison Expressions
        "=" = Eq([Id; 2]),
        "<>" = Neq([Id; 2]),
        "<" = Lt([Id; 2]),
        "<=" = Le([Id; 2]),
        ">" = Gt([Id; 2]),
        ">=" = Ge([Id; 2]),
        "between" = Between([Id; 3]),      // [expr, low, high]
        "in" = In([Id; 2]),                // [expr, list]
        "like" = Like([Id; 2]),            // [expr, pattern]

        // Logical Expressions
        "and" = And([Id; 2]),
        "or" = Or([Id; 2]),
        "not" = Not([Id; 1]),
        "is_null" = IsNull([Id; 1]),
        "is_not_null" = IsNotNull([Id; 1]),

        // Type Expressions
        "cast" = Cast([Id; 2]),            // [expr, type]
        "try_cast" = TryCast([Id; 2]),     // [expr, type]

        // Conditional Expressions
        "case" = Case([Id; 3]),            // [condition, then, else]
        "coalesce" = Coalesce(Box<[Id]>),  // [list of expressions]
        "nullif" = NullIf([Id; 2]),        // [left, right]

        // Aggregate Functions
        "count" = Count([Id; 1]),
        "count_distinct" = CountDistinct([Id; 1]),
        "sum" = Sum([Id; 1]),
        "avg" = Avg([Id; 1]),
        "min" = Min([Id; 1]),
        "max" = Max([Id; 1]),
        "stddev" = Stddev([Id; 1]),
        "variance" = Variance([Id; 1]),

        // Scalar Functions
        "abs" = Abs([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "pow" = Pow([Id; 2]),
        "floor" = Floor([Id; 1]),
        "ceil" = Ceil([Id; 1]),
        "round" = Round([Id; 2]),

        // String Functions
        "concat" = Concat([Id; 2]),
        "substring" = Substring([Id; 3]),
        "upper" = Upper([Id; 1]),
        "lower" = Lower([Id; 1]),
        "trim" = Trim([Id; 1]),
        "length" = Length([Id; 1]),

        // Date/Time Functions
        "date_part" = DatePart([Id; 2]),
        "date_trunc" = DateTrunc([Id; 2]),
        "now" = Now,

        // Window Functions
        "window" = Window([Id; 4]),        // [func, partition_by, order_by, frame]
        "row_number" = RowNumber,
        "rank" = Rank,
        "dense_rank" = DenseRank,
        "lag" = Lag([Id; 2]),
        "lead" = Lead([Id; 2]),
        "first_value" = FirstValue([Id; 1]),
        "last_value" = LastValue([Id; 1]),

        // Zarr-Specific Operators
        "zarr_scan" = ZarrScan([Id; 3]),   // [store_path, coordinates, variables]
        "coord_filter" = CoordFilter([Id; 3]), // [input, coord_name, range]
        "resample" = Resample([Id; 3]),    // [input, dimension, frequency]

        // List and Structure
        "list" = List(Box<[Id]>),
        "empty" = Empty,
        "alias" = Alias([Id; 2]),
        "sort_expr" = SortExpr([Id; 3]),   // [expr, asc, nulls_first]

        // Literals and References
        // Symbol variants (first one is the fallback for bare symbols when parsing)
        // Use format conventions to distinguish: columns are identifiers, literals are type:value
        Symbol(Symbol),                      // Universal symbol - used for columns, literals, tables, types

        // Boolean and null constants
        "true" = True,
        "false" = False,
        "null" = Null,
        "star" = Star,
    }
}

impl ZarrPlan {
    /// Check if this node is a relational operator
    pub fn is_relational(&self) -> bool {
        matches!(
            self,
            ZarrPlan::Scan(_)
                | ZarrPlan::Filter(_)
                | ZarrPlan::Project(_)
                | ZarrPlan::Aggregate(_)
                | ZarrPlan::Sort(_)
                | ZarrPlan::Limit(_)
                | ZarrPlan::Distinct(_)
                | ZarrPlan::Union(_)
                | ZarrPlan::InnerJoin(_)
                | ZarrPlan::LeftJoin(_)
                | ZarrPlan::RightJoin(_)
                | ZarrPlan::FullJoin(_)
                | ZarrPlan::CrossJoin(_)
                | ZarrPlan::SemiJoin(_)
                | ZarrPlan::AntiJoin(_)
                | ZarrPlan::ZarrScan(_)
                | ZarrPlan::CoordFilter(_)
                | ZarrPlan::Resample(_)
        )
    }

    /// Check if this node is an aggregate function
    pub fn is_aggregate(&self) -> bool {
        matches!(
            self,
            ZarrPlan::Count(_)
                | ZarrPlan::CountDistinct(_)
                | ZarrPlan::Sum(_)
                | ZarrPlan::Avg(_)
                | ZarrPlan::Min(_)
                | ZarrPlan::Max(_)
                | ZarrPlan::Stddev(_)
                | ZarrPlan::Variance(_)
        )
    }

    /// Check if this node is a comparison operator
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            ZarrPlan::Eq(_)
                | ZarrPlan::Neq(_)
                | ZarrPlan::Lt(_)
                | ZarrPlan::Le(_)
                | ZarrPlan::Gt(_)
                | ZarrPlan::Ge(_)
                | ZarrPlan::Between(_)
                | ZarrPlan::In(_)
                | ZarrPlan::Like(_)
        )
    }

    /// Check if this node is a logical operator
    pub fn is_logical(&self) -> bool {
        matches!(
            self,
            ZarrPlan::And(_)
                | ZarrPlan::Or(_)
                | ZarrPlan::Not(_)
                | ZarrPlan::IsNull(_)
                | ZarrPlan::IsNotNull(_)
        )
    }

    /// Check if this is a leaf node (no children)
    pub fn is_leaf(&self) -> bool {
        matches!(
            self,
            ZarrPlan::Symbol(_)
                | ZarrPlan::True
                | ZarrPlan::False
                | ZarrPlan::Null
                | ZarrPlan::Star
                | ZarrPlan::Empty
                | ZarrPlan::Now
                | ZarrPlan::RowNumber
                | ZarrPlan::Rank
                | ZarrPlan::DenseRank
        )
    }
}

/// Parse a literal symbol into its type and value
pub fn parse_literal(s: &Symbol) -> Option<(String, String)> {
    let s = s.as_str();
    let idx = s.find(':')?;
    let type_str = s[..idx].to_string();
    let value_str = s[idx + 1..].to_string();
    Some((type_str, value_str))
}

/// Create a literal symbol from type and value
pub fn make_literal(type_str: &str, value_str: &str) -> Symbol {
    format!("{}:{}", type_str, value_str).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::RecExpr;

    #[test]
    fn test_parse_simple_expression() {
        // Bare symbols like 'a' and 'b' are parsed as Symbol(Symbol)
        let expr: RecExpr<ZarrPlan> = "(+ a b)".parse().unwrap();
        assert_eq!(expr.as_ref().len(), 3);
    }

    #[test]
    fn test_parse_filter() {
        // t is table, x is column, i64:42 is literal (all parsed as Symbol)
        let expr: RecExpr<ZarrPlan> =
            "(filter (scan t empty) (= x i64:42))"
                .parse()
                .unwrap();
        assert!(expr.as_ref().len() > 0);
    }

    #[test]
    fn test_parse_aggregate() {
        let expr: RecExpr<ZarrPlan> =
            "(aggregate (scan t empty) empty (list (min x) (max y)))"
                .parse()
                .unwrap();
        assert!(expr.as_ref().len() > 0);
    }

    #[test]
    fn test_literal_parsing() {
        let lit = make_literal("f64", "3.14");
        let (t, v) = parse_literal(&lit).unwrap();
        assert_eq!(t, "f64");
        assert_eq!(v, "3.14");
    }

    #[test]
    fn test_is_relational() {
        let scan = ZarrPlan::Scan([Id::from(0), Id::from(1)]);
        assert!(scan.is_relational());

        let add = ZarrPlan::Add([Id::from(0), Id::from(1)]);
        assert!(!add.is_relational());
    }

    #[test]
    fn test_is_aggregate() {
        let min = ZarrPlan::Min([Id::from(0)]);
        assert!(min.is_aggregate());

        let add = ZarrPlan::Add([Id::from(0), Id::from(1)]);
        assert!(!add.is_aggregate());
    }
}
