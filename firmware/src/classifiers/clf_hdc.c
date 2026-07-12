#include <errno.h>
#include <limits.h>
#include <stdbool.h>
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
BUILD_ASSERT(CLF_CLASS_COUNT == 5U, "HDC rejection assumes four gestures plus null");

static uint8_t timestep_counts[HDC_DIM_BITS];
static uint16_t window_counts[HDC_DIM_BITS];
static uint32_t timestep_words[3][HDC_DIM_WORDS];
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

static bool tie_bit(const uint32_t words[HDC_DIM_WORDS], uint16_t bit_index)
{
	uint16_t word = bit_index / HDC_BITS_PER_WORD;
	uint8_t bit = bit_index % HDC_BITS_PER_WORD;

	return (words[word] & BIT(bit)) != 0U;
}

static void build_timestep_vector(const int16_t sample[CLF_CHANNELS], uint32_t out_words[HDC_DIM_WORDS])
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
			uint16_t bit_index = bit_base + bit;
			uint8_t count = timestep_counts[bit_index];

			if ((2U * count) > CLF_CHANNELS ||
			    ((2U * count) == CLF_CHANNELS && tie_bit(hdc_channel_tie_hv[0], bit_index))) {
				bundled |= BIT(bit);
			}
		}

		out_words[word] = bundled;
	}
}

static uint32_t permuted_word(const uint32_t words[HDC_DIM_WORDS], uint16_t output_word,
			      uint16_t rotation)
{
	uint16_t source_word = (output_word + HDC_DIM_WORDS - (rotation % HDC_DIM_WORDS)) %
			       HDC_DIM_WORDS;

	return words[source_word];
}

static void bundle_trigram(const uint32_t first[HDC_DIM_WORDS],
			   const uint32_t second[HDC_DIM_WORDS],
			   const uint32_t third[HDC_DIM_WORDS])
{
	for (uint16_t word = 0U; word < HDC_DIM_WORDS; word++) {
		uint32_t gram = permuted_word(first, word, 2U) ^
				permuted_word(second, word, 1U) ^
				third[word];

		add_word_to_u16_counts(window_counts, word, gram);
	}
}

static void build_query_vector(uint16_t n)
{
	for (uint16_t word = 0U; word < HDC_DIM_WORDS; word++) {
		uint16_t bit_base = word * HDC_BITS_PER_WORD;
		uint32_t bundled = 0U;

		for (uint8_t bit = 0U; bit < HDC_BITS_PER_WORD; bit++) {
			uint16_t bit_index = bit_base + bit;
			uint16_t count = window_counts[bit_index];

			if ((2U * count) > n ||
			    ((2U * count) == n && tie_bit(hdc_bundle_tie_hv[0], bit_index))) {
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
		build_timestep_vector(win[timestep], timestep_words[timestep % 3U]);
		if (timestep >= 2U) {
			bundle_trigram(timestep_words[(timestep - 2U) % 3U],
				       timestep_words[(timestep - 1U) % 3U],
				       timestep_words[timestep % 3U]);
		}
	}

	build_query_vector(n - 2U);

	/* Null is an open-set rejection decision. A single prototype cannot model
	 * heterogeneous free-living activity reliably, so only gesture prototypes
	 * compete in nearest-neighbor search. */
	for (uint8_t class_id = 0U; class_id < (CLF_CLASS_COUNT - 1U); class_id++) {
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
	if (best_distance > HDC_REJECTION_MAX_DISTANCE ||
	    (second_distance - best_distance) < HDC_REJECTION_MIN_MARGIN) {
		return CLF_CLASS_COUNT - 1U;
	}
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
