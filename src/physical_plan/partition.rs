use crate::reader::filter::CoordSelection;

// Serialize/Deserialize so the distributed codec can ship per-task partition
// subsets to workers. `CoordSelection` is plain ints, so serde handles it
// directly (no byte-DTO mirror needed, unlike ScalarValue-bearing types).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PartitionSpec {
    /// The selection this partition reads on the OUTER (axis-0) coordinate.
    /// Today the planner only emits chunk-aligned `Range`s; `Indices` becomes
    /// reachable once a resolved date-part selection is split across partitions.
    pub outer: CoordSelection,
}

impl PartitionSpec {
    /// A contiguous half-open `[start, end)` slice on the outer axis.
    pub fn range(start: u64, end: u64) -> Self {
        Self {
            outer: CoordSelection::Range(start as usize, end as usize),
        }
    }

    /// True when this partition reads nothing (used for uniform-length padding).
    pub fn is_empty(&self) -> bool {
        self.outer.is_empty()
    }

    /// `(start, end)` if the outer selection is a contiguous range.
    pub fn as_range(&self) -> Option<(usize, usize)> {
        self.outer.as_range()
    }
}

pub fn plan_partitions(
    outer_len: u64,
    chunk_len: u64,
    target_partitions: usize,
) -> Vec<PartitionSpec> {
    if outer_len == 0 {
        return vec![PartitionSpec::range(0, 0)];
    }

    let chunk_len = if chunk_len == 0 { outer_len } else { chunk_len };

    let n_chunks = outer_len.div_ceil(chunk_len);

    let p = (target_partitions as u64).min(n_chunks).max(1);
    let chunk_per_part = n_chunks.div_ceil(p);

    let mut specs = Vec::with_capacity(p as usize);

    let mut chunk_start = 0u64;
    while chunk_start < n_chunks {
        let chunk_end = (chunk_start + chunk_per_part).min(n_chunks);

        let outer_start = chunk_start * chunk_len;
        let outer_end = (chunk_end * chunk_len).min(outer_len);

        specs.push(PartitionSpec::range(outer_start, outer_end));
        chunk_start = chunk_end;
    }

    specs
}

/// Split a *resolved* coordinate selection (the surviving index set on the outer
/// axis, after the filter has been applied) into up to `target` contiguous,
/// chunk-aware pieces — one per output partition.
///
/// This is the surviving-set analogue of [`plan_partitions`]: instead of slicing
/// the full axis `[0, outer_len)` blind to the filter, it slices only what
/// survived, so a narrow filter (e.g. a single day out of 124 years) still fans
/// out across the cluster instead of collapsing onto one worker.
///
/// Invariants: the pieces are disjoint, their union equals `sel`, at least one
/// piece is returned, and — for the `Range` case — no underlying chunk is split
/// across two pieces (internal boundaries are chunk-aligned, so no chunk is read
/// twice). An empty selection yields a single empty piece.
///
/// NOTE: parallelism is capped by the number of surviving *chunks*, not `target`
/// — you can't split below a chunk without double-reading it. A surviving range
/// that lands in a single chunk yields a single piece regardless of `target`.
pub fn split_selection(sel: &CoordSelection, chunk_len: u64, target: usize) -> Vec<CoordSelection> {
    match sel {
        CoordSelection::Range(s, e) => split_range(*s, *e, chunk_len, target)
            .into_iter()
            .map(|(s, e)| CoordSelection::Range(s, e))
            .collect(),
        // Scattered positions (date-part filters): balanced contiguous groups by
        // count, but never splitting a single data-var chunk across two workers
        // (that chunk would then be read twice). For the common hourly case
        // (chunk_len == 1) every index is its own chunk, so this is the same
        // count-balanced split as before.
        CoordSelection::Indices(v) => split_indices(v, chunk_len, target)
            .into_iter()
            .map(CoordSelection::Indices)
            .collect(),
    }
}

/// Slice the surviving contiguous range `[s, e)` into chunk-aligned pieces.
///
/// Pieces break only on chunk boundaries (multiples of `chunk_len`) so a chunk is
/// never read by two workers; the first/last pieces are clipped to `[s, e)`.
fn split_range(s: usize, e: usize, chunk_len: u64, target: usize) -> Vec<(usize, usize)> {
    if e <= s {
        return vec![(s, s)];
    }
    // Unknown chunking => can't slice safely, so emit the whole range as one piece.
    if chunk_len == 0 {
        return vec![(s, e)];
    }
    let chunk_len = chunk_len as usize;

    let first_chunk = s / chunk_len;
    let last_chunk = (e - 1) / chunk_len;
    let n_chunks = last_chunk - first_chunk + 1;

    let p = target.min(n_chunks).max(1);
    let chunk_per_part = n_chunks.div_ceil(p);

    let mut out = Vec::with_capacity(p);
    let mut c = first_chunk;
    while c <= last_chunk {
        let c_end = (c + chunk_per_part).min(last_chunk + 1);
        let start = (c * chunk_len).max(s);
        let end = (c_end * chunk_len).min(e);
        out.push((start, end));
        c = c_end;
    }
    out
}

/// Split a sorted index list into up to `target` balanced contiguous groups,
/// keeping all survivors of one data-var chunk together.
///
/// Indices that share a chunk (`idx / chunk_len`) form an indivisible group, so a
/// chunk is never read by two workers. Groups are then packed greedily into ≤
/// `target` contiguous partitions of ~`ceil(len / p)` indices each. With
/// `chunk_len == 1` every index is its own group, recovering the previous
/// count-balanced split.
fn split_indices(v: &[usize], chunk_len: u64, target: usize) -> Vec<Vec<usize>> {
    if v.is_empty() {
        return vec![Vec::new()];
    }
    let chunk_len = if chunk_len == 0 {
        1
    } else {
        chunk_len as usize
    };

    // Consecutive runs of indices sharing a chunk (v is sorted ascending).
    let mut groups: Vec<&[usize]> = Vec::new();
    let mut start = 0;
    for i in 1..=v.len() {
        if i == v.len() || v[i] / chunk_len != v[start] / chunk_len {
            groups.push(&v[start..i]);
            start = i;
        }
    }

    let p = target.min(groups.len()).max(1);
    let per = v.len().div_ceil(p); // ~target indices per partition

    let mut out: Vec<Vec<usize>> = Vec::with_capacity(p);
    let mut cur: Vec<usize> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        cur.extend_from_slice(g);
        let groups_left = groups.len() - gi - 1;
        // Close the partition once it reaches the target size, but only while
        // enough groups remain to fill the partitions we still owe.
        if cur.len() >= per && out.len() + 1 < p && groups_left >= p - (out.len() + 1) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// An empty slice `[0, 0)` — reads nothing. Used to pad task groups so every
/// group has the same length.
fn empty_spec() -> PartitionSpec {
    PartitionSpec::range(0, 0)
}

/// Distribute `specs` across `task_count` distributed worker tasks.
///
/// Returns exactly `task_count` groups, **all of equal length** — a hard
/// requirement of `DistributedLeafExec`, whose per-task variants must all report
/// the same partition count. Uniformity is achieved by padding short groups with
/// empty specs (`empty_spec`), which read nothing.
///
/// The real specs are partitioned **contiguously and balanced** (the first
/// `len % task_count` groups get one extra), so each task reads an adjacent run
/// of chunks (good for sequential I/O) and their union is exactly the input —
/// disjoint and complete. Padding never duplicates or drops a real spec.
pub fn distribute_specs_across_tasks(
    specs: &[PartitionSpec],
    task_count: usize,
) -> Vec<Vec<PartitionSpec>> {
    let task_count = task_count.max(1);
    let p = specs.len();

    // Balanced contiguous sizes: first `rem` groups get `base + 1`, rest `base`.
    let base = p / task_count;
    let rem = p % task_count;
    // Uniform group length = the largest group size (ceil(p / task_count)).
    let k = if rem > 0 { base + 1 } else { base };

    let mut groups = Vec::with_capacity(task_count);
    let mut cursor = 0usize;
    for t in 0..task_count {
        let size = base + if t < rem { 1 } else { 0 };
        let mut group: Vec<PartitionSpec> = specs[cursor..cursor + size].to_vec();
        cursor += size;
        // Pad to the uniform length so all variants report the same count.
        group.resize_with(k, empty_spec);
        groups.push(group);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every valid partitioning must satisfy these invariants, regardless of
    /// inputs: at least one partition, starts at 0, reaches `outer_len`, and is
    /// contiguous + disjoint (each partition's start == the previous one's end).
    /// This *is* the correctness contract the whole feature rests on.
    fn assert_covers(specs: &[PartitionSpec], outer_len: u64) {
        assert!(!specs.is_empty(), "must emit at least one partition");
        let r = |s: &PartitionSpec| s.as_range().expect("planner emits ranges");
        assert_eq!(r(&specs[0]).0, 0, "must start at 0");
        assert_eq!(
            r(specs.last().unwrap()).1,
            outer_len as usize,
            "must reach the end"
        );
        for w in specs.windows(2) {
            assert_eq!(
                r(&w[0]).1,
                r(&w[1]).0,
                "gap or overlap between adjacent partitions"
            );
        }
    }

    /// Invariants for `distribute_specs_across_tasks`: exactly `task_count`
    /// groups, all the same length, and the non-empty specs across all groups
    /// equal the input exactly (disjoint + complete, no dup/drop).
    fn assert_distribution(
        groups: &[Vec<PartitionSpec>],
        input: &[PartitionSpec],
        task_count: usize,
    ) {
        assert_eq!(
            groups.len(),
            task_count,
            "must emit exactly task_count groups"
        );
        if let Some(first) = groups.first() {
            for g in groups {
                assert_eq!(g.len(), first.len(), "all groups must be the same length");
            }
        }
        // Collect non-empty (real) specs in order; must equal the input.
        let real: Vec<PartitionSpec> = groups
            .iter()
            .flatten()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        assert_eq!(real, input, "real specs must equal the input, in order");
    }

    #[test]
    fn distribute_even() {
        let specs = plan_partitions(6, 1, 6); // 6 specs
        let groups = distribute_specs_across_tasks(&specs, 3);
        assert_eq!(groups.len(), 3);
        assert_eq!(
            groups.iter().map(|g| g.len()).collect::<Vec<_>>(),
            vec![2, 2, 2]
        );
        assert_distribution(&groups, &specs, 3);
    }

    #[test]
    fn distribute_uneven_pads_to_uniform() {
        let specs = plan_partitions(7, 1, 7); // 7 specs across 3 tasks -> 3,2,2
        let groups = distribute_specs_across_tasks(&specs, 3);
        // All groups padded to length 3 (ceil(7/3)).
        for g in &groups {
            assert_eq!(g.len(), 3);
        }
        // Real counts are 3,2,2; groups 2 and 3 carry one empty pad each.
        assert_distribution(&groups, &specs, 3);
    }

    #[test]
    fn distribute_more_tasks_than_specs() {
        let specs = plan_partitions(2, 1, 2); // 2 specs, 4 tasks
        let groups = distribute_specs_across_tasks(&specs, 4);
        assert_eq!(groups.len(), 4);
        // ceil(2/4) = 1, so every group has length 1; two are empty pads.
        for g in &groups {
            assert_eq!(g.len(), 1);
        }
        assert_distribution(&groups, &specs, 4);
    }

    #[test]
    fn distribute_single_task_is_identity() {
        let specs = plan_partitions(5, 1, 5);
        let groups = distribute_specs_across_tasks(&specs, 1);
        assert_eq!(groups, vec![specs]);
    }

    #[test]
    fn even_split() {
        // 8 chunks of size 1, want 4 partitions -> 2 chunks each.
        let specs = plan_partitions(8, 1, 4);
        assert_eq!(specs.len(), 4);
        assert_eq!(specs[0], PartitionSpec::range(0, 2));
        assert_covers(&specs, 8);
    }

    #[test]
    fn uneven_split_remainder_rides_early_partitions() {
        // 7 chunks, want 4 -> ceil(7/4)=2 per part: [2,2,2,1].
        let specs = plan_partitions(7, 1, 4);
        assert_eq!(specs.len(), 4);
        assert_eq!(
            specs
                .iter()
                .map(|s| {
                    let (a, b) = s.as_range().unwrap();
                    b - a
                })
                .collect::<Vec<_>>(),
            vec![2, 2, 2, 1]
        );
        assert_covers(&specs, 7);
    }

    #[test]
    fn fewer_chunks_than_target() {
        // Only 3 chunks but 8 requested: never more partitions than chunks.
        let specs = plan_partitions(3, 1, 8);
        assert_eq!(specs.len(), 3);
        assert_covers(&specs, 3);
    }

    #[test]
    fn larger_chunks_group_correctly() {
        // 10 elems, chunk size 5 -> 2 chunks. target 4 capped to 2.
        let specs = plan_partitions(10, 5, 4);
        assert_eq!(specs.len(), 2);
        assert_eq!(
            specs,
            vec![PartitionSpec::range(0, 5), PartitionSpec::range(5, 10)]
        );
        assert_covers(&specs, 10);
    }

    #[test]
    fn partial_last_chunk_is_clamped() {
        // 7 elems, chunk 2 -> 4 chunks, last is half-full. End must clamp to 7, not 8.
        let specs = plan_partitions(7, 2, 4);
        assert_eq!(specs.last().unwrap().as_range().unwrap().1, 7);
        assert_covers(&specs, 7);
    }

    #[test]
    fn single_partition_when_target_is_one() {
        let specs = plan_partitions(100, 10, 1);
        assert_eq!(specs, vec![PartitionSpec::range(0, 100)]);
    }

    #[test]
    fn missing_chunk_len_falls_back_to_one_partition() {
        // chunk_len 0 means "unknown" -> treat whole axis as one chunk.
        let specs = plan_partitions(100, 0, 8);
        assert_eq!(specs.len(), 1);
        assert_covers(&specs, 100);
    }

    #[test]
    fn empty_array_yields_one_empty_partition() {
        let specs = plan_partitions(0, 1, 4);
        assert_eq!(specs, vec![PartitionSpec::range(0, 0)]);
    }

    #[test]
    fn target_zero_still_yields_one_partition() {
        // Defensive: target 0 must not produce zero partitions (max(1) guard).
        let specs = plan_partitions(10, 1, 0);
        assert_eq!(specs.len(), 1);
        assert_covers(&specs, 10);
    }

    // ── split_selection ────────────────────────────────────────────────────

    fn range(s: usize, e: usize) -> CoordSelection {
        CoordSelection::Range(s, e)
    }
    fn indices(v: &[usize]) -> CoordSelection {
        CoordSelection::Indices(v.to_vec())
    }

    /// The split of a `Range(s, e)` must be disjoint, cover exactly `[s, e)`, and
    /// break only on chunk boundaries (no chunk read by two pieces).
    fn assert_range_split(pieces: &[CoordSelection], s: usize, e: usize, chunk_len: usize) {
        assert!(!pieces.is_empty(), "must emit at least one piece");
        let rs: Vec<(usize, usize)> = pieces.iter().map(|p| p.as_range().unwrap()).collect();
        assert_eq!(rs.first().unwrap().0, s, "must start at s");
        assert_eq!(rs.last().unwrap().1, e, "must reach e");
        for w in rs.windows(2) {
            assert_eq!(w[0].1, w[1].0, "gap or overlap between pieces");
            assert_eq!(
                w[0].1 % chunk_len,
                0,
                "internal boundary must be chunk-aligned"
            );
        }
    }

    #[test]
    fn split_range_hourly_even() {
        // 24 hourly chunks across 3 workers -> 8 each. The single-day fix.
        let pieces = split_selection(&range(0, 24), 1, 3);
        assert_eq!(pieces, vec![range(0, 8), range(8, 16), range(16, 24)]);
        assert_range_split(&pieces, 0, 24, 1);
    }

    #[test]
    fn split_range_offset_window() {
        // A day deep into the axis (e.g. 2023 in an hours-since-1900 store).
        let pieces = split_selection(&range(1_086_000, 1_086_024), 1, 3);
        assert_eq!(pieces.len(), 3);
        assert_range_split(&pieces, 1_086_000, 1_086_024, 1);
    }

    #[test]
    fn split_range_clipped_to_chunk_boundaries() {
        // [13, 47) over chunk_len 10 touches chunks 1..=4. Pieces clip at the
        // ends (13, 47) but break only at 20/30/40.
        let pieces = split_selection(&range(13, 47), 10, 4);
        assert_eq!(
            pieces,
            vec![range(13, 20), range(20, 30), range(30, 40), range(40, 47)]
        );
        assert_range_split(&pieces, 13, 47, 10);
    }

    #[test]
    fn split_range_single_chunk_cannot_split() {
        // 24 indices but chunk_len 100 => one chunk => one piece, regardless of
        // target. Documents the chunk-granularity parallelism ceiling.
        let pieces = split_selection(&range(0, 24), 100, 3);
        assert_eq!(pieces, vec![range(0, 24)]);
    }

    #[test]
    fn split_range_fewer_chunks_than_target() {
        // 3 surviving chunks, 8 workers => never more than 3 pieces.
        let pieces = split_selection(&range(0, 3), 1, 8);
        assert_eq!(pieces, vec![range(0, 1), range(1, 2), range(2, 3)]);
    }

    #[test]
    fn split_range_empty_yields_one_empty() {
        let pieces = split_selection(&range(5, 5), 1, 3);
        assert_eq!(pieces, vec![range(5, 5)]);
    }

    #[test]
    fn split_range_unknown_chunk_is_one_piece() {
        // chunk_len 0 ("unknown") => whole surviving range treated as one chunk.
        let pieces = split_selection(&range(10, 40), 0, 4);
        assert_eq!(pieces, vec![range(10, 40)]);
    }

    #[test]
    fn split_range_target_zero_yields_one_piece() {
        let pieces = split_selection(&range(0, 10), 1, 0);
        assert_eq!(pieces, vec![range(0, 10)]);
    }

    #[test]
    fn split_indices_even() {
        let pieces = split_selection(&indices(&[0, 5, 10, 15, 20, 25]), 1, 3);
        assert_eq!(
            pieces,
            vec![indices(&[0, 5]), indices(&[10, 15]), indices(&[20, 25])]
        );
    }

    #[test]
    fn split_indices_remainder() {
        // 5 indices across 3 -> ceil(5/3)=2 per group -> [2,2,1].
        let pieces = split_selection(&indices(&[1, 2, 3, 4, 5]), 1, 3);
        assert_eq!(
            pieces,
            vec![indices(&[1, 2]), indices(&[3, 4]), indices(&[5])]
        );
    }

    #[test]
    fn split_indices_fewer_than_target() {
        let pieces = split_selection(&indices(&[7]), 1, 4);
        assert_eq!(pieces, vec![indices(&[7])]);
    }

    #[test]
    fn split_indices_empty_yields_one_empty() {
        let pieces = split_selection(&indices(&[]), 1, 3);
        assert_eq!(pieces, vec![indices(&[])]);
    }

    #[test]
    fn split_indices_keeps_a_shared_chunk_together() {
        // 0 and 5 share chunk 0 (len 10): they must NOT be split across workers,
        // even though target=2 — that chunk would be read twice. One partition.
        let pieces = split_selection(&indices(&[0, 5]), 10, 2);
        assert_eq!(pieces, vec![indices(&[0, 5])]);
    }

    #[test]
    fn split_indices_splits_on_chunk_boundaries() {
        // chunk 0: {0,1,2}, chunk 3: {30}. Groups stay whole across 2 workers.
        let pieces = split_selection(&indices(&[0, 1, 2, 30]), 10, 2);
        assert_eq!(pieces, vec![indices(&[0, 1, 2]), indices(&[30])]);
    }

    #[test]
    fn split_indices_chunk_aware_union_equals_input() {
        // chunks of len 10: {3,7},{12},{25,28},{41}. Union is preserved and no
        // chunk's indices are spread across two pieces.
        let input = [3usize, 7, 12, 25, 28, 41];
        let pieces = split_selection(&indices(&input), 10, 3);
        let rebuilt: Vec<usize> = pieces
            .iter()
            .flat_map(|p| match p {
                CoordSelection::Indices(v) => v.clone(),
                _ => panic!("expected Indices"),
            })
            .collect();
        assert_eq!(rebuilt, input);
        // Each chunk id appears in exactly one piece.
        for piece in &pieces {
            if let CoordSelection::Indices(v) = piece {
                let chunks: std::collections::HashSet<usize> = v.iter().map(|&i| i / 10).collect();
                for other in &pieces {
                    if std::ptr::eq(piece, other) {
                        continue;
                    }
                    if let CoordSelection::Indices(o) = other {
                        for &i in o {
                            assert!(!chunks.contains(&(i / 10)), "chunk split across pieces");
                        }
                    }
                }
            }
        }
    }

    /// The union of all pieces must reconstruct the input exactly — the
    /// disjoint+complete contract the distributed scan relies on.
    #[test]
    fn split_indices_union_equals_input() {
        let input = [2usize, 9, 14, 27, 30, 41, 55];
        let pieces = split_selection(&indices(&input), 1, 3);
        let rebuilt: Vec<usize> = pieces
            .iter()
            .flat_map(|p| match p {
                CoordSelection::Indices(v) => v.clone(),
                _ => panic!("expected Indices"),
            })
            .collect();
        assert_eq!(rebuilt, input);
    }
}
