#include <limits.h>
#include <stdint.h>

#include "benchmark.h"

struct classifier_benchmark_stats g_classifier_benchmark_stats = {
	.min_cycles = UINT32_MAX,
};

void classifier_benchmark_record(uint32_t cycles)
{
	g_classifier_benchmark_stats.inference_count++;
	g_classifier_benchmark_stats.total_cycles += cycles;
	if (cycles < g_classifier_benchmark_stats.min_cycles) {
		g_classifier_benchmark_stats.min_cycles = cycles;
	}
	if (cycles > g_classifier_benchmark_stats.max_cycles) {
		g_classifier_benchmark_stats.max_cycles = cycles;
	}
}
