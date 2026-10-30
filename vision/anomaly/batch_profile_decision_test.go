package anomaly

import (
	"math"
	"testing"
)

func TestEvaluateBatchProfileDecisionsUsesFixedP2Gate(t *testing.T) {
	metrics := []BatchProfileMetric{
		{Mode: BatchProfileSequential, BatchSize: 1, WallP95MS: 20, FramesPerSecond: 50},
		{Mode: BatchProfileTensorBatch, BatchSize: 2, WallP95MS: 18, FramesPerSecond: 100},
		{Mode: BatchProfileTensorBatch, BatchSize: 4, WallP95MS: 22, FramesPerSecond: 125},
	}

	decisions := evaluateBatchProfileDecisions(metrics)
	if len(decisions) != 2 {
		t.Fatalf("decisions = %d, want 2", len(decisions))
	}

	first := decisions[0]
	if first.CandidateBatchSize != 2 || !first.LatencyWithinBaseline || !first.ThroughputGateMet || !first.EligibleForP2PathReview {
		t.Fatalf("batch 2 decision unexpectedly rejected: %+v", first)
	}
	if math.Abs(first.ThroughputRatio-2) > 1e-9 || first.MinimumThroughputRatio != 2 {
		t.Fatalf("unexpected batch 2 throughput gate: %+v", first)
	}

	second := decisions[1]
	if second.CandidateBatchSize != 4 || second.LatencyWithinBaseline || second.EligibleForP2PathReview {
		t.Fatalf("batch 4 decision unexpectedly accepted: %+v", second)
	}
	if !second.ThroughputGateMet || math.Abs(second.ThroughputRatio-2.5) > 1e-9 {
		t.Fatalf("unexpected batch 4 throughput result: %+v", second)
	}
	if second.Reasons[0] != "p95_latency_exceeds_serial_baseline" || second.Reasons[1] != "throughput_ratio_meets_2x" {
		t.Fatalf("unexpected rejection reasons: %+v", second.Reasons)
	}
}

func TestEvaluateBatchProfileDecisionsWithoutBaseline(t *testing.T) {
	if decisions := evaluateBatchProfileDecisions(nil); decisions == nil || len(decisions) != 0 {
		t.Fatalf("decisions = %#v, want empty non-nil slice", decisions)
	}
}
