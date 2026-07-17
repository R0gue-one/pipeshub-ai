import type { NotificationListItem } from './api';

const MAX_RECORDED_LATENCIES = 1_000;
const deliveryLatencyByNotificationId = new Map<string, number>();

interface ThroughputBenchmark {
  total: number;
  startedAtMs: number;
  receivedSequences: Set<number>;
  firstArrivalMs: number;
  lastArrivalMs: number;
  totalLatencyMs: number;
}

const throughputBenchmarks = new Map<string, ThroughputBenchmark>();

function withLatency(
  item: NotificationListItem,
  latencyMs: number | undefined,
): NotificationListItem {
  if (latencyMs === undefined) return item;

  return {
    ...item,
    message: `${item.message ?? ''} [latency: ${latencyMs}ms]`.trim(),
  };
}

function recordThroughputBenchmark(item: NotificationListItem, latencyMs: number): void {
  const benchmarkId = item.payload?.benchmark_id;
  const sequence = item.payload?.benchmark_sequence;
  const total = item.payload?.benchmark_total;
  const startedAtMs = item.payload?.benchmark_started_at_ms;
  if (
    typeof benchmarkId !== 'string' ||
    typeof sequence !== 'number' ||
    typeof total !== 'number' ||
    typeof startedAtMs !== 'number'
  ) {
    return;
  }

  const arrivalMs = Date.now();
  let benchmark = throughputBenchmarks.get(benchmarkId);
  if (!benchmark) {
    benchmark = {
      total,
      startedAtMs,
      receivedSequences: new Set<number>(),
      firstArrivalMs: arrivalMs,
      lastArrivalMs: arrivalMs,
      totalLatencyMs: 0,
    };
    throughputBenchmarks.set(benchmarkId, benchmark);
  }

  if (benchmark.receivedSequences.has(sequence)) return;

  benchmark.receivedSequences.add(sequence);
  benchmark.lastArrivalMs = arrivalMs;
  benchmark.totalLatencyMs += latencyMs;

  const received = benchmark.receivedSequences.size;
  if (received === 1) {
    console.info('[notifications] Throughput benchmark started', {
      benchmarkId,
      expected: total,
    });
  }

  if (received !== benchmark.total) return;

  const deliveryWindowMs = Math.max(1, benchmark.lastArrivalMs - benchmark.firstArrivalMs);
  const completionMs = Math.max(1, benchmark.lastArrivalMs - benchmark.startedAtMs);
  const deliveryPerSecond =
    benchmark.total === 1 ? 0 : ((benchmark.total - 1) * 1000) / deliveryWindowMs;
  const endToEndPerSecond = (benchmark.total * 1000) / completionMs;

  console.info('[notifications] Throughput benchmark completed', {
    benchmarkId,
    received,
    deliveryWindowMs,
    deliveryPerSecond: Number(deliveryPerSecond.toFixed(2)),
    deliveryPerMinute: Number((deliveryPerSecond * 60).toFixed(2)),
    endToEndCompletionMs: completionMs,
    endToEndPerSecond: Number(endToEndPerSecond.toFixed(2)),
    averageLatencyMs: Number((benchmark.totalLatencyMs / received).toFixed(2)),
  });
  throughputBenchmarks.delete(benchmarkId);
}

/** TESTING ONLY: measure latency exactly once when a live socket event arrives. */
export function recordNotificationDeliveryLatency(
  item: NotificationListItem,
): NotificationListItem {
  const sendTimeMs = item.payload?.send_time_ms;
  if (!item._id || typeof sendTimeMs !== 'number') return item;

  const latencyMs = Math.max(0, Date.now() - sendTimeMs);
  deliveryLatencyByNotificationId.set(item._id, latencyMs);

  if (deliveryLatencyByNotificationId.size > MAX_RECORDED_LATENCIES) {
    const oldestId = deliveryLatencyByNotificationId.keys().next().value;
    if (oldestId) deliveryLatencyByNotificationId.delete(oldestId);
  }

  console.info('[notifications] WebSocket delivery latency', {
    notificationId: item._id,
    latencyMs,
  });
  recordThroughputBenchmark(item, latencyMs);

  return withLatency(item, latencyMs);
}

/**
 * Preserve a previously measured socket latency when a panel refresh replaces
 * the in-memory notification with the REST representation.
 */
export function tagNotificationWithRecordedLatency(
  item: NotificationListItem,
): NotificationListItem {
  return withLatency(item, deliveryLatencyByNotificationId.get(item._id));
}
