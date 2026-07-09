#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/sys/util.h>

#include "classifier.h"
#include "generated/hdc_memories.h"

#define HDC_BITS_PER_WORD 32U
#define HDC_DIM_BITS (HDC_DIM_WORDS * HDC_BITS_PER_WORD)

BUILD_ASSERT(HDC_DIM_WORDS == 64U, "HDC dimension must be 2048 bits");
BUILD_ASSERT(HDC_DIM_BITS == 2048U, "HDC dimension must be 2048 bits");
BUILD_ASSERT(HDC_CHANNEL_COUNT == CLF_CHANNELS, "HDC channel count mismatch");
BUILD_ASSERT(HDC_CLASS_VECTOR_COUNT == CLF_CLASS_COUNT, "HDC class count mismatch");

static uint8_t timestep_counts[HDC_DIM_BITS];
static uint16_t window_counts[HDC_DIM_BITS];
static uint32_t timestep_words[HDC_DIM_WORDS];
static uint32_t query_words[HDC_DIM_WORDS];
static int16_t last_score;

static uint8_t hdc_level_index(uint8_t channel, int16_t value)
{
	int32_t lo = hdc_level_min[channel];
	int32_t hi = hdc_level_max[channel];
	int32_t span = MAX(hi - lo, 1);
	int32_t shifted = (int32_t)value - lo;
	int32_t index = (shifted * HDC_LEVEL_COUNT) / span;

	if (index < 0) {
		index = 0;
	} else if (index >= HDC_LEVEL_COUNT) {
		index = HDC_LEVEL_COUNT - 1U;
	}

	return (uint8_t)index;
}

static void add_word_to_u8_counts(uint8_t *counts, uint16_t word_index, uint32_t word)
{
	uint16_t bit_base = word_index * HDC_BITS_PER_WORD;

	for (uint8_t bit = 0U; bit < HDC_BITS_PER_WORD; bit++) {
		if ((word & BIT(bit)) != 0U) {
			counts[bit_base + bit]++;
		}
	}
}

static void add_word_to_u16_counts(uint16_t *counts, uint16_t word_index, uint32_t word)
{
	uint16_t bit_base = word_index * HDC_BITS_PER_WORD;

	for (uint8_t bit = 0U; bit < HDC_BITS_PER_WORD; bit++) {
		if ((word & BIT(bit)) != 0U) {
			counts[bit_base + bit]++;
		}
	}
}

static void build_timestep_vector(const int16_t sample[CLF_CHANNELS])
{
	memset(timestep_counts, 0, sizeof(timestep_counts));

	for (uint8_t channel = 0U; channel < CLF_CHANNELS; channel++) {
		uint8_t level = hdc_level_index(channel, sample[channel]);

		for (uint16_t word = 0U; word < HDC_DIM_WORDS; word++) {
			uint32_t bound = hdc_level_hv[level][word] ^ hdc_channel_hv[channel][word];

			add_word_to_u8_counts(timestep_counts, word, bound);
		}
	}

	for (uint16_t word = 0U; word < HDC_DIM_WORDS; word++) {
		uint16_t bit_base = word * HDC_BITS_PER_WORD;
		uint32_t bundled = 0U;

		for (uint8_t bit = 0U; bit < HDC_BITS_PER_WORD; bit++) {
			if (timestep_counts[bit_base + bit] > (CLF_CHANNELS / 2U)) {
				bundled |= BIT(bit);
			}
		}

		timestep_words[word] = bundled;
	}
}

static void bundle_permuted_timestep(uint16_t timestep)
{
	uint16_t rotation = timestep % HDC_DIM_WORDS;

	for (uint16_t word = 0U; word < HDC_DIM_WORDS; word++) {
		uint16_t rotated_word = (word + rotation) % HDC_DIM_WORDS;

		add_word_to_u16_counts(window_counts, rotated_word, timestep_words[word]);
	}
}

static void build_query_vector(uint16_t n)
{
	for (uint16_t word = 0U; word < HDC_DIM_WORDS; word++) {
		uint16_t bit_base = word * HDC_BITS_PER_WORD;
		uint32_t bundled = 0U;

		for (uint8_t bit = 0U; bit < HDC_BITS_PER_WORD; bit++) {
			if (window_counts[bit_base + bit] > (n / 2U)) {
				bundled |= BIT(bit);
			}
		}

		query_words[word] = bundled;
	}
}

static uint16_t hamming_distance_to_class(uint8_t class_id)
{
	uint16_t distance = 0U;

	for (uint16_t word = 0U; word < HDC_DIM_WORDS; word++) {
		distance += (uint16_t)__builtin_popcount(query_words[word] ^ hdc_class_hv[class_id][word]);
	}

	return distance;
}

int clf_init(void)
{
	last_score = 0;
	return 0;
}

int clf_process_window(const int16_t (*win)[CLF_CHANNELS], uint16_t n)
{
	uint16_t best_distance = UINT16_MAX;
	uint16_t second_distance = UINT16_MAX;
	uint8_t best_class = 0U;

	if (win == NULL || n != CLF_WINDOW_SAMPLES) {
		return -EINVAL;
	}

	memset(window_counts, 0, sizeof(window_counts));

	for (uint16_t timestep = 0U; timestep < n; timestep++) {
		build_timestep_vector(win[timestep]);
		bundle_permuted_timestep(timestep);
	}

	build_query_vector(n);

	for (uint8_t class_id = 0U; class_id < CLF_CLASS_COUNT; class_id++) {
		uint16_t distance = hamming_distance_to_class(class_id);

		if (distance < best_distance) {
			second_distance = best_distance;
			best_distance = distance;
			best_class = class_id;
		} else if (distance < second_distance) {
			second_distance = distance;
		}
	}

	last_score = (int16_t)(second_distance - best_distance);
	return best_class;
}

const char *clf_name(void)
{
	return "hdc";
}

int16_t clf_last_score(void)
{
	return last_score;
}
