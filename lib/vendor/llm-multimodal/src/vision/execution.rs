//! Shared execution primitives for CPU-bound vision preprocessing.
//!
//! This module defines task granularity without owning a process-wide thread
//! pool. Rayon remains responsible for worker lifecycle and work stealing.

const PARALLEL_MIN_BYTES: usize = 1 << 19;
const MAX_TASKS_PER_OPERATION: usize = 8;

pub(crate) fn scope<'scope, OP, R>(operation: OP) -> R
where
    OP: FnOnce(&rayon::Scope<'scope>) -> R + Send,
    R: Send,
{
    rayon::scope(operation)
}

pub(crate) fn task_count(
    output_bytes: usize,
    work_items: usize,
    min_items_per_task: usize,
) -> usize {
    debug_assert!(min_items_per_task > 0);
    if output_bytes < PARALLEL_MIN_BYTES || work_items < 2 * min_items_per_task {
        return 1;
    }

    (work_items / min_items_per_task)
        .min(rayon::current_num_threads())
        .clamp(1, MAX_TASKS_PER_OPERATION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_operations_stay_serial() {
        assert_eq!(task_count(PARALLEL_MIN_BYTES - 1, 1_024, 1), 1);
        assert_eq!(task_count(PARALLEL_MIN_BYTES, 63, 32), 1);
    }

    #[test]
    fn task_count_respects_operation_and_executor_limits() {
        let tasks = task_count(PARALLEL_MIN_BYTES, usize::MAX, 1);
        assert!(tasks <= MAX_TASKS_PER_OPERATION);
        assert!(tasks <= rayon::current_num_threads());
    }
}
