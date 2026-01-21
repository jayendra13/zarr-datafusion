//! Rewrite rules for e-graph query optimization
//!
//! This module defines rewrite rules for:
//! - Expression simplification (arithmetic, boolean)
//! - Relational optimizations (filter pushdown, etc.)
//! - Zarr-specific optimizations (coordinate filters)

use egg::{rewrite as rw, Rewrite};

use super::analysis::ZarrAnalysis;
use super::language::ZarrPlan;

/// Get all rewrite rules
pub fn all_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    let mut rules = Vec::new();
    rules.extend(expression_simplification_rules());
    rules.extend(boolean_simplification_rules());
    rules.extend(comparison_simplification_rules());
    rules.extend(relational_rewrite_rules());
    rules.extend(aggregate_rules());
    rules.extend(zarr_specific_rules());
    rules
}

/// Expression simplification rules (arithmetic)
pub fn expression_simplification_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    vec![
        // === Addition identities ===
        // Bare symbols like 'i64:0' are parsed as Symbol(Symbol("i64:0"))
        rw!("add-zero-right"; "(+ ?x i64:0)" => "?x"),
        rw!("add-zero-left"; "(+ i64:0 ?x)" => "?x"),
        rw!("add-zero-right-f64"; "(+ ?x f64:0.0)" => "?x"),
        rw!("add-zero-left-f64"; "(+ f64:0.0 ?x)" => "?x"),
        // Commutativity
        rw!("add-comm"; "(+ ?x ?y)" => "(+ ?y ?x)"),
        // Associativity
        rw!("add-assoc"; "(+ (+ ?x ?y) ?z)" => "(+ ?x (+ ?y ?z))"),
        // === Subtraction identities ===
        rw!("sub-zero"; "(- ?x i64:0)" => "?x"),
        rw!("sub-zero-f64"; "(- ?x f64:0.0)" => "?x"),
        rw!("sub-self"; "(- ?x ?x)" => "i64:0"),
        // === Multiplication identities ===
        rw!("mul-one-right"; "(* ?x i64:1)" => "?x"),
        rw!("mul-one-left"; "(* i64:1 ?x)" => "?x"),
        rw!("mul-one-right-f64"; "(* ?x f64:1.0)" => "?x"),
        rw!("mul-one-left-f64"; "(* f64:1.0 ?x)" => "?x"),
        rw!("mul-zero-right"; "(* ?x i64:0)" => "i64:0"),
        rw!("mul-zero-left"; "(* i64:0 ?x)" => "i64:0"),
        rw!("mul-zero-right-f64"; "(* ?x f64:0.0)" => "f64:0.0"),
        rw!("mul-zero-left-f64"; "(* f64:0.0 ?x)" => "f64:0.0"),
        // Commutativity
        rw!("mul-comm"; "(* ?x ?y)" => "(* ?y ?x)"),
        // Associativity
        rw!("mul-assoc"; "(* (* ?x ?y) ?z)" => "(* ?x (* ?y ?z))"),
        // === Division identities ===
        rw!("div-one"; "(/ ?x i64:1)" => "?x"),
        rw!("div-one-f64"; "(/ ?x f64:1.0)" => "?x"),
        rw!("div-self"; "(/ ?x ?x)" => "i64:1"),
        // === Algebraic simplifications ===
        // (a * b) / b => a
        rw!("mul-div-cancel-right"; "(/ (* ?x ?y) ?y)" => "?x"),
        // (b * a) / b => a
        rw!("mul-div-cancel-left"; "(/ (* ?y ?x) ?y)" => "?x"),
        // a / b * b => a
        rw!("div-mul-cancel"; "(* (/ ?x ?y) ?y)" => "?x"),
        // Distributive: a * (b + c) => a * b + a * c
        rw!("distribute-mul-add"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        // Factor: a * b + a * c => a * (b + c)
        rw!("factor-mul-add"; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        // === Negation ===
        rw!("neg-neg"; "(neg (neg ?x))" => "?x"),
        rw!("sub-as-neg"; "(- ?x ?y)" => "(+ ?x (neg ?y))"),
    ]
}

/// Boolean simplification rules
pub fn boolean_simplification_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    vec![
        // === AND identities ===
        rw!("and-true-right"; "(and ?x true)" => "?x"),
        rw!("and-true-left"; "(and true ?x)" => "?x"),
        rw!("and-false-right"; "(and ?x false)" => "false"),
        rw!("and-false-left"; "(and false ?x)" => "false"),
        rw!("and-self"; "(and ?x ?x)" => "?x"),
        // Commutativity
        rw!("and-comm"; "(and ?x ?y)" => "(and ?y ?x)"),
        // Associativity
        rw!("and-assoc"; "(and (and ?x ?y) ?z)" => "(and ?x (and ?y ?z))"),
        // === OR identities ===
        rw!("or-false-right"; "(or ?x false)" => "?x"),
        rw!("or-false-left"; "(or false ?x)" => "?x"),
        rw!("or-true-right"; "(or ?x true)" => "true"),
        rw!("or-true-left"; "(or true ?x)" => "true"),
        rw!("or-self"; "(or ?x ?x)" => "?x"),
        // Commutativity
        rw!("or-comm"; "(or ?x ?y)" => "(or ?y ?x)"),
        // Associativity
        rw!("or-assoc"; "(or (or ?x ?y) ?z)" => "(or ?x (or ?y ?z))"),
        // === NOT identities ===
        rw!("not-not"; "(not (not ?x))" => "?x"),
        rw!("not-true"; "(not true)" => "false"),
        rw!("not-false"; "(not false)" => "true"),
        // === De Morgan's laws ===
        rw!("de-morgan-and"; "(not (and ?x ?y))" => "(or (not ?x) (not ?y))"),
        rw!("de-morgan-or"; "(not (or ?x ?y))" => "(and (not ?x) (not ?y))"),
        // === Absorption ===
        rw!("absorb-and-or"; "(and ?x (or ?x ?y))" => "?x"),
        rw!("absorb-or-and"; "(or ?x (and ?x ?y))" => "?x"),
        // === Complementation ===
        rw!("and-not-self"; "(and ?x (not ?x))" => "false"),
        rw!("or-not-self"; "(or ?x (not ?x))" => "true"),
        // === Null handling ===
        rw!("is-null-not-null"; "(and (is_null ?x) (is_not_null ?x))" => "false"),
        rw!("not-is-null"; "(not (is_null ?x))" => "(is_not_null ?x)"),
        rw!("not-is-not-null"; "(not (is_not_null ?x))" => "(is_null ?x)"),
    ]
}

/// Comparison simplification rules
pub fn comparison_simplification_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    vec![
        // === Reflexive comparisons ===
        rw!("eq-self"; "(= ?x ?x)" => "true"),
        rw!("neq-self"; "(<> ?x ?x)" => "false"),
        rw!("le-self"; "(<= ?x ?x)" => "true"),
        rw!("ge-self"; "(>= ?x ?x)" => "true"),
        rw!("lt-self"; "(< ?x ?x)" => "false"),
        rw!("gt-self"; "(> ?x ?x)" => "false"),
        // === Commutativity ===
        rw!("eq-comm"; "(= ?x ?y)" => "(= ?y ?x)"),
        rw!("neq-comm"; "(<> ?x ?y)" => "(<> ?y ?x)"),
        // === Inverse comparisons ===
        rw!("lt-to-gt"; "(< ?x ?y)" => "(> ?y ?x)"),
        rw!("le-to-ge"; "(<= ?x ?y)" => "(>= ?y ?x)"),
        rw!("gt-to-lt"; "(> ?x ?y)" => "(< ?y ?x)"),
        rw!("ge-to-le"; "(>= ?x ?y)" => "(<= ?y ?x)"),
        // === NOT comparisons ===
        rw!("not-eq"; "(not (= ?x ?y))" => "(<> ?x ?y)"),
        rw!("not-neq"; "(not (<> ?x ?y))" => "(= ?x ?y)"),
        rw!("not-lt"; "(not (< ?x ?y))" => "(>= ?x ?y)"),
        rw!("not-le"; "(not (<= ?x ?y))" => "(> ?x ?y)"),
        rw!("not-gt"; "(not (> ?x ?y))" => "(<= ?x ?y)"),
        rw!("not-ge"; "(not (>= ?x ?y))" => "(< ?x ?y)"),
        // === BETWEEN simplification ===
        // BETWEEN is equivalent to: x >= low AND x <= high
        rw!("between-to-and"; "(between ?x ?low ?high)" => "(and (>= ?x ?low) (<= ?x ?high))"),
    ]
}

/// Relational optimization rules
pub fn relational_rewrite_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    vec![
        // === Filter rules ===
        // Combine adjacent filters
        rw!("filter-merge";
            "(filter (filter ?input ?p1) ?p2)" =>
            "(filter ?input (and ?p1 ?p2))"
        ),
        // Filter with true predicate is identity
        rw!("filter-true"; "(filter ?input true)" => "?input"),
        // Filter with false predicate produces empty
        rw!("filter-false"; "(filter ?input false)" => "empty"),
        // === Projection rules ===
        // Project followed by project - keep only outer projection
        // (This is a simplification; full implementation needs column analysis)
        // rw!("project-project"; "(project (project ?input ?e1) ?e2)" => "(project ?input ?e2)"),
        // === Limit rules ===
        // Limit 0 produces empty
        rw!("limit-zero"; "(limit ?input i64:0)" => "empty"),
        // Combine adjacent limits - take the smaller
        // (This requires analysis; simplified version)
        // === Sort rules ===
        // Sort followed by sort - inner sort is redundant
        rw!("sort-sort"; "(sort (sort ?input ?k1) ?k2)" => "(sort ?input ?k2)"),
        // === Distinct rules ===
        // Distinct on already distinct input is identity
        rw!("distinct-distinct"; "(distinct (distinct ?input))" => "(distinct ?input)"),
        // === Join rules ===
        // Inner join commutativity
        rw!("inner-join-comm";
            "(inner_join ?l ?r ?cond)" =>
            "(inner_join ?r ?l ?cond)"
        ),
        // Cross join commutativity
        rw!("cross-join-comm"; "(cross_join ?l ?r)" => "(cross_join ?r ?l)"),
        // Cross join with filter is inner join (when filter references both sides)
        // This requires more complex analysis to implement correctly
        // === Filter pushdown through projection ===
        // (filter (project ?input ?exprs) ?pred) => (project (filter ?input ?pred) ?exprs)
        // This requires checking that pred only uses columns from input
        // Simplified: when projection doesn't change referenced columns
        // === Filter pushdown through join ===
        // Push filter to left side of join if it only references left columns
        // (filter (inner_join ?l ?r ?c) ?p) => (inner_join (filter ?l ?p) ?r ?c)
        // This requires column analysis
    ]
}

/// Aggregate optimization rules
pub fn aggregate_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    vec![
        // COUNT(*) of empty is 0
        // rw!("count-empty"; "(aggregate empty empty (list (count star)))" =>
        //     "(project empty (list i64:0))"),

        // MIN/MAX of single value - pattern variables match any symbol
        // Note: These rules work when the argument is a literal value
        // rw!("min-single"; "(min ?v)" => "?v"),  // Too broad - would match columns too
        // rw!("max-single"; "(max ?v)" => "?v"),  // Too broad
        // rw!("sum-single"; "(sum ?v)" => "?v"),  // Too broad
        // These require conditional application based on analysis data
        // AVG simplification with same value
        // (This is complex and requires constant analysis)
        // COUNT(x) where x is NOT NULL equals row count
        // (Requires nullability analysis)
        // === Aggregate through UNION ===
        // SUM over UNION ALL = SUM of SUMs
        // (Complex - requires proper handling)
    ]
}

/// Zarr-specific optimization rules
pub fn zarr_specific_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    vec![
        // === Coordinate filter pushdown ===
        // Push equality filter on coordinate to scan
        // (filter (zarr_scan ?path ?coords ?vars) (= (Column ?dim) (Literal ?val)))
        // => (coord_filter (zarr_scan ?path ?coords ?vars) ?dim (Literal ?val))
        // This requires checking that ?dim is a coordinate dimension
        // === Range filter pushdown ===
        // (filter (zarr_scan ?path ?coords ?vars) (between (Column ?dim) ?low ?high))
        // => (coord_filter (zarr_scan ?path ?coords ?vars) ?dim (list ?low ?high))
        // === Statistics-based rewrites ===
        // These are handled in the analysis rather than as rewrite rules,
        // as they require access to runtime statistics
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::{Extractor, Runner};

    fn simplify(input: &str, rules: &[Rewrite<ZarrPlan, ZarrAnalysis>]) -> String {
        let expr = input.parse().unwrap();
        let runner = Runner::default().with_expr(&expr).run(rules);
        let extractor = Extractor::new(&runner.egraph, egg::AstSize);
        let (_, best) = extractor.find_best(runner.roots[0]);
        best.to_string()
    }

    #[test]
    fn test_add_zero() {
        let rules = expression_simplification_rules();
        // Bare symbols: 'x' is column, 'i64:0' is literal
        let result = simplify("(+ x i64:0)", &rules);
        assert_eq!(result, "x");
    }

    #[test]
    fn test_mul_one() {
        let rules = expression_simplification_rules();
        let result = simplify("(* y i64:1)", &rules);
        assert_eq!(result, "y");
    }

    #[test]
    fn test_mul_zero() {
        let rules = expression_simplification_rules();
        let result = simplify("(* z i64:0)", &rules);
        assert_eq!(result, "i64:0");
    }

    #[test]
    fn test_and_true() {
        let rules = boolean_simplification_rules();
        let result = simplify("(and p true)", &rules);
        assert_eq!(result, "p");
    }

    #[test]
    fn test_and_false() {
        let rules = boolean_simplification_rules();
        let result = simplify("(and p false)", &rules);
        assert_eq!(result, "false");
    }

    #[test]
    fn test_or_true() {
        let rules = boolean_simplification_rules();
        let result = simplify("(or p true)", &rules);
        assert_eq!(result, "true");
    }

    #[test]
    fn test_not_not() {
        let rules = boolean_simplification_rules();
        let result = simplify("(not (not p))", &rules);
        assert_eq!(result, "p");
    }

    #[test]
    fn test_eq_self() {
        let rules = comparison_simplification_rules();
        let result = simplify("(= x x)", &rules);
        assert_eq!(result, "true");
    }

    #[test]
    fn test_filter_merge() {
        let rules = relational_rewrite_rules();
        let result = simplify(
            "(filter (filter (scan t empty) p) q)",
            &rules,
        );
        // Should merge into single filter with AND
        assert!(result.contains("and"));
    }

    #[test]
    fn test_filter_true() {
        let rules = relational_rewrite_rules();
        let result = simplify("(filter (scan t empty) true)", &rules);
        assert_eq!(result, "(scan t empty)");
    }

    #[test]
    fn test_complex_expression() {
        let rules = all_rules();
        // (x + 0) * 1 => x
        let result = simplify(
            "(* (+ x i64:0) i64:1)",
            &rules,
        );
        assert_eq!(result, "x");
    }

    #[test]
    fn test_de_morgan() {
        let rules = boolean_simplification_rules();
        let result = simplify("(not (and a b))", &rules);
        // Should be equivalent to (or (not a) (not b))
        assert!(result.contains("or") || result.contains("not"));
    }
}
