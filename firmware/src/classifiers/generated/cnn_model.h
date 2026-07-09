#ifndef GENERATED_CNN_MODEL_H
#define GENERATED_CNN_MODEL_H

#include <stdint.h>

/*
 * CNN export intentionally blocked.
 *
 * The latest 120 Hz model quantized successfully, but PTQ reduced macro-F1 from
 * 0.327733 to 0.263471, a 0.064262 absolute gap. The project requirement is
 * <= 0.01, so this header must stay non-deployable until QAT, calibration, or
 * more data brings the int8 gap under the threshold.
 */
static const float g_cnn_input_mean[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float g_cnn_input_std[6] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
static const uint8_t g_cnn_model[1] = { 0x00 };
static const unsigned int g_cnn_model_len = 0U;

#warning "CNN int8 export blocked: PTQ macro-F1 gap exceeds 0.01"

#endif
