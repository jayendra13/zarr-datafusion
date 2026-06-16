// Serialize/Deserialize so the distributed codec can ship per-task partition
// subsets to workers. PartitionSpec is plain u64s, so serde handles it directly
// (no byte-DTO mirror needed, unlike ScalarValue-bearing types).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PartitionSpec {
    pub outer_start: u64,
    pub outer_end: u64,
}

pub fn plan_partitions(
    outer_len: u64,
    chunk_len: u64,
    target_partitions: usize,
) -> Vec<PartitionSpec> {
    if outer_len == 0 {
        return vec![PartitionSpec {
            outer_start: 0,
            outer_end: 0,
        }];
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

        specs.push(PartitionSpec {
            outer_start,
            outer_end,
        });
        chunk_start = chunk_end;
    }

    specs
}

/// An empty slice `[0, 0)` — reads nothing. Used to pad task groups so every
/// group has the same length.
fn empty_spec() -> PartitionSpec {
    PartitionSpec {
        outer_start: 0,
        outer_end: 0,
    }
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
        assert_eq!(specs[0].outer_start, 0, "must start at 0");
        assert_eq!(
            specs.last().unwrap().outer_end,
            outer_len,
            "must reach the end"
        );
        for w in specs.windows(2) {
            assert_eq!(
                w[0].outer_end, w[1].outer_start,
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
            .filter(|s| s.outer_end > s.outer_start)
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
        assert_eq!(
            specs[0],
            PartitionSpec {
                outer_start: 0,
                outer_end: 2
            }
        );
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
                .map(|s| s.outer_end - s.outer_start)
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
            vec![
                PartitionSpec {
                    outer_start: 0,
                    outer_end: 5
                },
                PartitionSpec {
                    outer_start: 5,
                    outer_end: 10
                },
            ]
        );
        assert_covers(&specs, 10);
    }

    #[test]
    fn partial_last_chunk_is_clamped() {
        // 7 elems, chunk 2 -> 4 chunks, last is half-full. End must clamp to 7, not 8.
        let specs = plan_partitions(7, 2, 4);
        assert_eq!(specs.last().unwrap().outer_end, 7);
        assert_covers(&specs, 7);
    }

    #[test]
    fn single_partition_when_target_is_one() {
        let specs = plan_partitions(100, 10, 1);
        assert_eq!(
            specs,
            vec![PartitionSpec {
                outer_start: 0,
                outer_end: 100
            }]
        );
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
        assert_eq!(
            specs,
            vec![PartitionSpec {
                outer_start: 0,
                outer_end: 0
            }]
        );
    }

    #[test]
    fn target_zero_still_yields_one_partition() {
        // Defensive: target 0 must not produce zero partitions (max(1) guard).
        let specs = plan_partitions(10, 1, 0);
        assert_eq!(specs.len(), 1);
        assert_covers(&specs, 10);
    }
}
