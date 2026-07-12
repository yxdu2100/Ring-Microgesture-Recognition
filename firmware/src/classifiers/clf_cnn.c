#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include "classifier.h"
#include "generated/cnn_model.h"

LOG_MODULE_REGISTER(clf_cnn, LOG_LEVEL_INF);

#if __has_include(<tensorflow/lite/micro/micro_interpreter.h>)

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace {

constexpr int kTensorArenaSize = 40 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model *model;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input_tensor;
TfLiteTensor *output_tensor;
bool ready;
int16_t last_score;

float raw_to_physical(uint8_t channel, int16_t sample)
{
	if (channel < 3U) {
		return ((float)sample * 8.0f) / 32768.0f;
	}

	return ((float)sample * 2000.0f) / 32768.0f;
}

int8_t quantize_raw_sample(uint8_t channel, int16_t sample)
{
	float physical = raw_to_physical(channel, sample);
	float standardized = (physical - g_cnn_input_mean[channel]) / g_cnn_input_std[channel];
	int32_t q = lroundf(standardized / input_tensor->params.scale) + input_tensor->params.zero_point;

	if (q < INT8_MIN) {
		q = INT8_MIN;
	} else if (q > INT8_MAX) {
		q = INT8_MAX;
	}

	return static_cast<int8_t>(q);
}

int32_t output_value(size_t index)
{
	switch (output_tensor->type) {
	case kTfLiteInt8:
		return output_tensor->data.int8[index];
	case kTfLiteUInt8:
		return static_cast<int32_t>(output_tensor->data.uint8[index]) -
		       output_tensor->params.zero_point;
	case kTfLiteInt16:
		return output_tensor->data.i16[index];
	default:
		return INT32_MIN;
	}
}

}  // namespace

extern "C" int clf_init(void)
{
	if (g_cnn_model_len == 0U) {
		LOG_WRN("CNN model placeholder is empty");
		return -ENOENT;
	}

	model = tflite::GetModel(g_cnn_model);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		LOG_ERR("CNN model schema %d != supported %d",
			model->version(), TFLITE_SCHEMA_VERSION);
		return -EINVAL;
	}

	static tflite::MicroMutableOpResolver<6> resolver;
	if (resolver.AddConv2D() != kTfLiteOk ||
	    resolver.AddFullyConnected() != kTfLiteOk ||
	    resolver.AddReshape() != kTfLiteOk ||
	    resolver.AddExpandDims() != kTfLiteOk ||
	    resolver.AddMean() != kTfLiteOk ||
	    resolver.AddSoftmax() != kTfLiteOk) {
		LOG_ERR("Failed to register CNN TFLM ops");
		return -EINVAL;
	}

	static tflite::MicroInterpreter static_interpreter(model, resolver,
							   tensor_arena,
							   kTensorArenaSize);
	interpreter = &static_interpreter;

	if (interpreter->AllocateTensors() != kTfLiteOk) {
		LOG_ERR("CNN tensor allocation failed");
		return -ENOMEM;
	}

	input_tensor = interpreter->input(0);
	output_tensor = interpreter->output(0);
	if (input_tensor == nullptr || output_tensor == nullptr) {
		return -EINVAL;
	}

	if (input_tensor->type != kTfLiteInt8) {
		LOG_ERR("CNN input tensor must be int8");
		return -ENOTSUP;
	}

	ready = true;
	last_score = 0;
	LOG_INF("CNN classifier initialized with %d byte tensor arena", kTensorArenaSize);
	return 0;
}

extern "C" int clf_process_window(const int16_t (*win)[CLF_CHANNELS], uint16_t n)
{
	size_t needed = (size_t)CLF_WINDOW_SAMPLES * CLF_CHANNELS;
	int32_t best = INT32_MIN;
	int32_t second = INT32_MIN;
	int best_class = 0;

	if (!ready) {
		return -EAGAIN;
	}

	if (win == nullptr || n != CLF_WINDOW_SAMPLES) {
		return -EINVAL;
	}

	if (input_tensor->bytes < needed) {
		return -EMSGSIZE;
	}

	for (uint16_t sample = 0U; sample < CLF_WINDOW_SAMPLES; sample++) {
		for (uint8_t channel = 0U; channel < CLF_CHANNELS; channel++) {
			input_tensor->data.int8[(sample * CLF_CHANNELS) + channel] =
				quantize_raw_sample(channel, win[sample][channel]);
		}
	}

	if (interpreter->Invoke() != kTfLiteOk) {
		LOG_WRN("CNN invoke failed");
		return -EAGAIN;
	}

	for (uint8_t class_id = 0U; class_id < CLF_CLASS_COUNT; class_id++) {
		int32_t value = output_value(class_id);

		if (value > best) {
			second = best;
			best = value;
			best_class = class_id;
		} else if (value > second) {
			second = value;
		}
	}

	last_score = (int16_t)CLAMP(best - second, INT16_MIN, INT16_MAX);
	return best_class;
}

extern "C" const char *clf_name(void)
{
	return "cnn";
}

extern "C" int16_t clf_last_score(void)
{
	return last_score;
}

#else

#warning "TFLM headers not found; CNN classifier skeleton will not run until the tflite-micro module is installed"

int clf_init(void)
{
	LOG_ERR("TFLM headers are unavailable");
	return -ENOSYS;
}

int clf_process_window(const int16_t (*win)[CLF_CHANNELS], uint16_t n)
{
	ARG_UNUSED(win);
	ARG_UNUSED(n);

	return -EAGAIN;
}

const char *clf_name(void)
{
	return "cnn";
}

int16_t clf_last_score(void)
{
	return 0;
}

#endif
