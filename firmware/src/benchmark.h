#ifndef RING_BENCHMARK_H
#define RING_BENCHMARK_H

#include <stdint.h>

struct classifier_benchmark_stats {
	volatile uint32_t inference_count;
	volatile uint64_t total_cycles;
	volatile uint32_t min_cycles;
	volatile uint32_t max_cycles;
	uint32_t cycles_per_second;
};

extern struct classifier_benchmark_stats g_classifier_benchmark_stats;

void classifier_benchmark_record(uint32_t cycles);

#endif
