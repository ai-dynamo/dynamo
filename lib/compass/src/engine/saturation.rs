// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::types::QueueTrend;

#[derive(Debug, Clone)]
pub struct SaturationResult {
    pub component: String,
    pub utilization: f64,
    pub queue_trend: QueueTrend,
    pub saturation_score: f64,
}

#[derive(Debug, Clone)]
pub struct QueueSnapshot {
    pub component: String,
    pub timestamps: Vec<f64>,
    pub queue_lengths: Vec<f64>,
    pub arrival_rates: Vec<f64>,
    pub capacity: f64,
}

pub fn detect_saturation(snapshots: &[QueueSnapshot]) -> Vec<SaturationResult> {
    snapshots.iter().map(analyze_component).collect()
}

fn analyze_component(snapshot: &QueueSnapshot) -> SaturationResult {
    let avg_queue_length = mean(&snapshot.queue_lengths);
    let avg_arrival_rate = mean(&snapshot.arrival_rates);

    let utilization = if snapshot.capacity > 0.0 {
        (avg_queue_length / snapshot.capacity).min(1.0)
    } else {
        0.0
    };

    let queue_trend = detect_trend(&snapshot.queue_lengths);

    let growth_rate = match queue_trend {
        QueueTrend::Growing => compute_growth_rate(&snapshot.queue_lengths),
        QueueTrend::Draining => -compute_growth_rate(&snapshot.queue_lengths).abs(),
        QueueTrend::Stable => 0.0,
    };

    let saturation_score = (utilization * (1.0 + growth_rate.max(0.0))).min(1.0);

    let _ = avg_arrival_rate;

    SaturationResult {
        component: snapshot.component.clone(),
        utilization,
        queue_trend,
        saturation_score,
    }
}

fn detect_trend(values: &[f64]) -> QueueTrend {
    if values.len() < 3 {
        return QueueTrend::Stable;
    }

    let slope = linear_regression_slope(values);
    let avg = mean(values);
    let normalized_slope = if avg > 0.0 { slope / avg } else { slope };

    if normalized_slope > 0.05 {
        QueueTrend::Growing
    } else if normalized_slope < -0.05 {
        QueueTrend::Draining
    } else {
        QueueTrend::Stable
    }
}

fn linear_regression_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = mean(values);

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator.abs() < 1e-12 {
        0.0
    } else {
        numerator / denominator
    }
}

fn compute_growth_rate(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let first_half = mean(&values[..values.len() / 2]);
    let second_half = mean(&values[values.len() / 2..]);
    if first_half > 0.0 {
        (second_half - first_half) / first_half
    } else if second_half > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_queue() {
        let snapshot = QueueSnapshot {
            component: "prefill.gpu".to_string(),
            timestamps: (0..10).map(|i| i as f64).collect(),
            queue_lengths: vec![5.0; 10],
            arrival_rates: vec![10.0; 10],
            capacity: 10.0,
        };
        let result = analyze_component(&snapshot);
        assert_eq!(result.queue_trend, QueueTrend::Stable);
        assert!((result.utilization - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_growing_queue() {
        let snapshot = QueueSnapshot {
            component: "kvbm.radix_tree_lock".to_string(),
            timestamps: (0..10).map(|i| i as f64).collect(),
            queue_lengths: (1..=10).map(|i| i as f64 * 2.0).collect(),
            arrival_rates: vec![10.0; 10],
            capacity: 20.0,
        };
        let result = analyze_component(&snapshot);
        assert_eq!(result.queue_trend, QueueTrend::Growing);
        assert!(result.saturation_score > 0.5);
    }

    #[test]
    fn test_draining_queue() {
        let snapshot = QueueSnapshot {
            component: "nixl.h2d_channel".to_string(),
            timestamps: (0..10).map(|i| i as f64).collect(),
            queue_lengths: (0..10).rev().map(|i| i as f64 * 2.0).collect(),
            arrival_rates: vec![5.0; 10],
            capacity: 20.0,
        };
        let result = analyze_component(&snapshot);
        assert_eq!(result.queue_trend, QueueTrend::Draining);
    }

    #[test]
    fn test_saturated_component() {
        let snapshot = QueueSnapshot {
            component: "kvbm.radix_tree_lock".to_string(),
            timestamps: (0..20).map(|i| i as f64).collect(),
            queue_lengths: (0..20).map(|i| 8.0 + i as f64 * 0.5).collect(),
            arrival_rates: vec![10.0; 20],
            capacity: 10.0,
        };
        let result = analyze_component(&snapshot);
        assert!(result.utilization > 0.8);
        assert!(result.saturation_score > 0.8);
    }

    #[test]
    fn test_detect_saturation_multiple() {
        let snapshots = vec![
            QueueSnapshot {
                component: "a".to_string(),
                timestamps: vec![0.0, 1.0],
                queue_lengths: vec![5.0, 5.0],
                arrival_rates: vec![10.0, 10.0],
                capacity: 10.0,
            },
            QueueSnapshot {
                component: "b".to_string(),
                timestamps: vec![0.0, 1.0],
                queue_lengths: vec![9.0, 10.0],
                arrival_rates: vec![10.0, 10.0],
                capacity: 10.0,
            },
        ];
        let results = detect_saturation(&snapshots);
        assert_eq!(results.len(), 2);
        assert!(results[1].utilization > results[0].utilization);
    }
}
