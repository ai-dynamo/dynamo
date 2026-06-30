use super::*;

#[test]
fn test_simulation_worker_addition() {
    // Test that adding a worker causes minimal churn with HRW vs Random
    let num_workers: usize = 3;
    let num_loras: usize = 4;
    let slots_per_worker: usize = 4;
    let load_per_lora: usize = 5;

    // --- HRW ---
    let hrw_churn = {
        let alloc_config = LoraAllocationConfig {
            enabled: true,
            algorithm: AllocationAlgorithmType::Hrw,
            timestep_secs: 1,
            scale_down_cooldown_ticks: 0,
            rate_window_multiplier: 5,
            ..Default::default()
        };
        let routing_table = LoraRoutingTable::new();
        let state_tracker = LoraStateTracker::new();
        let load_estimator = Arc::new(LoadEstimator::new());
        let mut controller = LoraController::new(
            alloc_config,
            routing_table.clone(),
            state_tracker.clone(),
            load_estimator.clone(),
        );

        for i in 0..num_workers {
            let w = WorkerWithDpRank::new(i as u64, 0);
            state_tracker.set_worker_capacity(w, slots_per_worker as u32);
        }
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            for _ in 0..load_per_lora {
                load_estimator.increment_load(&name);
            }
        }
        controller.recompute_now();

        let mut before: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                before.insert(lora_name, config.replica_set);
            }
        }

        // Add a new worker
        state_tracker.set_worker_capacity(
            WorkerWithDpRank::new(num_workers as u64, 0),
            slots_per_worker as u32,
        );
        controller.recompute_now();

        let mut after: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                after.insert(lora_name, config.replica_set);
            }
        }

        let (loads, unloads) = compute_churn(&before, &after);
        (loads, unloads)
    };

    // --- Random ---
    let random_churn = {
        let workers: Vec<WorkerWithDpRank> = (0..num_workers)
            .map(|i| WorkerWithDpRank::new(i as u64, 0))
            .collect();
        let total_slots = num_workers * slots_per_worker;
        let total_load = num_loras * load_per_lora;
        let mut rng = RandomAllocator::new(42);

        let mut worker_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = workers
            .iter()
            .map(|&w| (w, (0, slots_per_worker)))
            .collect();

        // Allocate before
        let mut before: AllocationSnapshot = HashMap::new();
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            let fraction = load_per_lora as f64 / total_load as f64;
            let desired = (fraction * total_slots as f64)
                .ceil()
                .max(1.0)
                .min(workers.len() as f64) as usize;
            let set = rng.compute_replica_set(&name, &workers, desired, &worker_slot_usage);
            for w in &set {
                if let Some((used, _)) = worker_slot_usage.get_mut(w) {
                    *used += 1;
                }
            }
            before.insert(name, set);
        }

        // Add new worker
        let mut new_workers = workers.clone();
        new_workers.push(WorkerWithDpRank::new(num_workers as u64, 0));
        let new_total_slots = (num_workers + 1) * slots_per_worker;
        let mut new_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = new_workers
            .iter()
            .map(|&w| (w, (0, slots_per_worker)))
            .collect();

        // Reallocate after
        let mut after: AllocationSnapshot = HashMap::new();
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            let fraction = load_per_lora as f64 / total_load as f64;
            let desired = (fraction * new_total_slots as f64)
                .ceil()
                .max(1.0)
                .min(new_workers.len() as f64) as usize;
            let set = rng.compute_replica_set(&name, &new_workers, desired, &new_slot_usage);
            for w in &set {
                if let Some((used, _)) = new_slot_usage.get_mut(w) {
                    *used += 1;
                }
            }
            after.insert(name, set);
        }

        let (loads, unloads) = compute_churn(&before, &after);
        (loads, unloads)
    };

    // --- MCF ---
    let mcf_churn = {
        let alloc_config = LoraAllocationConfig {
            enabled: true,
            algorithm: AllocationAlgorithmType::MinCostFlow,
            timestep_secs: 1,
            scale_down_cooldown_ticks: 0,
            rate_window_multiplier: 5,
            ..Default::default()
        };
        let routing_table = LoraRoutingTable::new();
        let state_tracker = LoraStateTracker::new();
        let load_estimator = Arc::new(LoadEstimator::new());
        let mut controller = LoraController::new(
            alloc_config,
            routing_table.clone(),
            state_tracker.clone(),
            load_estimator.clone(),
        );

        for i in 0..num_workers {
            let w = WorkerWithDpRank::new(i as u64, 0);
            state_tracker.set_worker_capacity(w, slots_per_worker as u32);
        }
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            for _ in 0..load_per_lora {
                load_estimator.increment_load(&name);
            }
        }
        controller.recompute_now();

        let mut before: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                before.insert(lora_name, config.replica_set);
            }
        }

        // Add a new worker
        state_tracker.set_worker_capacity(
            WorkerWithDpRank::new(num_workers as u64, 0),
            slots_per_worker as u32,
        );
        controller.recompute_now();

        let mut after: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                after.insert(lora_name, config.replica_set);
            }
        }

        let (loads, unloads) = compute_churn(&before, &after);
        (loads, unloads)
    };

    let hrw_total = hrw_churn.0 + hrw_churn.1;
    let random_total = random_churn.0 + random_churn.1;
    let mcf_total = mcf_churn.0 + mcf_churn.1;

    println!("\n{}", "=".repeat(72));
    println!(
        "Worker Addition Churn (3 → 4 workers, {} LoRAs, {} slots/worker)",
        num_loras, slots_per_worker
    );
    println!("{}", "=".repeat(72));
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Metric", "HRW", "Random", "MCF"
    );
    println!("  {:->12}  {:->8}  {:->8}  {:->8}", "", "", "", "");
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Loads", hrw_churn.0, random_churn.0, mcf_churn.0
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Unloads", hrw_churn.1, random_churn.1, mcf_churn.1
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Total", hrw_total, random_total, mcf_total
    );
    println!();

    // HRW should have bounded churn
    assert!(
        hrw_total <= num_loras * 2,
        "HRW worker addition churn ({}) should be bounded by 2x num_loras ({})",
        hrw_total,
        num_loras * 2
    );
    // MCF should have bounded churn
    assert!(
        mcf_total <= num_loras * 2,
        "MCF worker addition churn ({}) should be bounded by 2x num_loras ({})",
        mcf_total,
        num_loras * 2
    );
}
#[test]
fn test_simulation_worker_removal() {
    // Test that removing a worker causes minimal churn with HRW vs Random vs MCF
    let num_workers: usize = 5;
    let num_loras: usize = 6;
    let slots_per_worker: usize = 4;
    let load_per_lora: usize = 3;
    let removed_worker_id: u64 = 2;

    // --- HRW ---
    let hrw_churn = {
        let alloc_config = LoraAllocationConfig {
            enabled: true,
            algorithm: AllocationAlgorithmType::Hrw,
            timestep_secs: 1,
            scale_down_cooldown_ticks: 0,
            rate_window_multiplier: 5,
            ..Default::default()
        };
        let routing_table = LoraRoutingTable::new();
        let state_tracker = LoraStateTracker::new();
        let load_estimator = Arc::new(LoadEstimator::new());
        let mut controller = LoraController::new(
            alloc_config,
            routing_table.clone(),
            state_tracker.clone(),
            load_estimator.clone(),
        );

        for i in 0..num_workers {
            let w = WorkerWithDpRank::new(i as u64, 0);
            state_tracker.set_worker_capacity(w, slots_per_worker as u32);
        }
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            for _ in 0..load_per_lora {
                load_estimator.increment_load(&name);
            }
        }
        controller.recompute_now();

        let mut before: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                before.insert(lora_name, config.replica_set);
            }
        }

        // Remove worker
        state_tracker.handle_worker_removal(WorkerWithDpRank::new(removed_worker_id, 0));
        controller.recompute_now();

        let mut after: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                after.insert(lora_name, config.replica_set);
            }
        }

        // Verify removed worker is not in any replica set
        for (lora_name, workers) in &after {
            for w in workers {
                assert!(
                    w.worker_id != removed_worker_id,
                    "Removed worker {} should not be in replica set for {}",
                    removed_worker_id,
                    lora_name
                );
            }
        }

        compute_churn(&before, &after)
    };

    // --- Random ---
    let random_churn = {
        let workers: Vec<WorkerWithDpRank> = (0..num_workers)
            .map(|i| WorkerWithDpRank::new(i as u64, 0))
            .collect();
        let total_slots = num_workers * slots_per_worker;
        let total_load = num_loras * load_per_lora;
        let mut rng = RandomAllocator::new(42);

        let mut worker_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = workers
            .iter()
            .map(|&w| (w, (0, slots_per_worker)))
            .collect();

        // Allocate before
        let mut before: AllocationSnapshot = HashMap::new();
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            let fraction = load_per_lora as f64 / total_load as f64;
            let desired = (fraction * total_slots as f64)
                .ceil()
                .max(1.0)
                .min(workers.len() as f64) as usize;
            let set = rng.compute_replica_set(&name, &workers, desired, &worker_slot_usage);
            for w in &set {
                if let Some((used, _)) = worker_slot_usage.get_mut(w) {
                    *used += 1;
                }
            }
            before.insert(name, set);
        }

        // Remove worker — allocate on reduced set
        let remaining_workers: Vec<WorkerWithDpRank> = workers
            .iter()
            .filter(|w| w.worker_id != removed_worker_id)
            .copied()
            .collect();
        let new_total_slots = remaining_workers.len() * slots_per_worker;
        let mut new_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = remaining_workers
            .iter()
            .map(|&w| (w, (0, slots_per_worker)))
            .collect();

        let mut after: AllocationSnapshot = HashMap::new();
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            let fraction = load_per_lora as f64 / total_load as f64;
            let desired = (fraction * new_total_slots as f64)
                .ceil()
                .max(1.0)
                .min(remaining_workers.len() as f64) as usize;
            let set = rng.compute_replica_set(&name, &remaining_workers, desired, &new_slot_usage);
            for w in &set {
                if let Some((used, _)) = new_slot_usage.get_mut(w) {
                    *used += 1;
                }
            }
            after.insert(name, set);
        }

        compute_churn(&before, &after)
    };

    // --- MCF ---
    let mcf_churn = {
        let alloc_config = LoraAllocationConfig {
            enabled: true,
            algorithm: AllocationAlgorithmType::MinCostFlow,
            timestep_secs: 1,
            scale_down_cooldown_ticks: 0,
            rate_window_multiplier: 5,
            ..Default::default()
        };
        let routing_table = LoraRoutingTable::new();
        let state_tracker = LoraStateTracker::new();
        let load_estimator = Arc::new(LoadEstimator::new());
        let mut controller = LoraController::new(
            alloc_config,
            routing_table.clone(),
            state_tracker.clone(),
            load_estimator.clone(),
        );

        for i in 0..num_workers {
            let w = WorkerWithDpRank::new(i as u64, 0);
            state_tracker.set_worker_capacity(w, slots_per_worker as u32);
        }
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            for _ in 0..load_per_lora {
                load_estimator.increment_load(&name);
            }
        }
        controller.recompute_now();

        let mut before: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                before.insert(lora_name, config.replica_set);
            }
        }

        // Remove worker
        state_tracker.handle_worker_removal(WorkerWithDpRank::new(removed_worker_id, 0));
        controller.recompute_now();

        let mut after: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                after.insert(lora_name, config.replica_set);
            }
        }

        // Verify removed worker is not in any replica set
        for (lora_name, workers) in &after {
            for w in workers {
                assert!(
                    w.worker_id != removed_worker_id,
                    "MCF: Removed worker {} should not be in replica set for {}",
                    removed_worker_id,
                    lora_name
                );
            }
        }

        compute_churn(&before, &after)
    };

    let hrw_total = hrw_churn.0 + hrw_churn.1;
    let random_total = random_churn.0 + random_churn.1;
    let mcf_total = mcf_churn.0 + mcf_churn.1;

    println!("\n{}", "=".repeat(72));
    println!(
        "Worker Removal Churn ({} → {} workers, {} LoRAs, {} slots/worker)",
        num_workers,
        num_workers - 1,
        num_loras,
        slots_per_worker
    );
    println!("{}", "=".repeat(72));
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Metric", "HRW", "Random", "MCF"
    );
    println!("  {:->12}  {:->8}  {:->8}  {:->8}", "", "", "", "");
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Loads", hrw_churn.0, random_churn.0, mcf_churn.0
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Unloads", hrw_churn.1, random_churn.1, mcf_churn.1
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Total", hrw_total, random_total, mcf_total
    );
    println!();

    // HRW should have bounded churn
    assert!(
        hrw_total <= num_loras * 2,
        "HRW worker removal churn ({}) should be bounded by 2x num_loras ({})",
        hrw_total,
        num_loras * 2
    );
    // MCF should have bounded churn
    assert!(
        mcf_total <= num_loras * 2,
        "MCF worker removal churn ({}) should be bounded by 2x num_loras ({})",
        mcf_total,
        num_loras * 2
    );
}
