#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include "ble/ble.h"
#include "classifiers/classifier.h"
#include "modules/imu.h"

LOG_MODULE_REGISTER(app, LOG_LEVEL_INF);

#define CONTROL_EVENTS (EVENT_START_STREAMING | EVENT_ENTER_LOW_POWER)

#if !defined(CONFIG_CLASSIFIER_NONE) && !defined(CONFIG_CLASSIFIER_MLC)
#define CLASSIFIER_HOP_SAMPLES 64U
#define CLASSIFIER_QUEUE_LEN 128U

K_MSGQ_DEFINE(classifier_sample_msgq,
	      sizeof(struct bt_app_imu_sample),
	      CLASSIFIER_QUEUE_LEN,
	      4);

static int16_t classifier_ring[CLF_WINDOW_SAMPLES][CLF_CHANNELS];
static int16_t classifier_window[CLF_WINDOW_SAMPLES][CLF_CHANNELS];
static uint16_t ring_write_index;
static uint16_t ring_count;
static uint16_t hop_count;
static bool sample_id_valid;
static uint16_t last_sample_id16;
static uint32_t unwrapped_sample_id;
static uint32_t window_end_sample_id;

static void classifier_reset(void)
{
	k_msgq_purge(&classifier_sample_msgq);
	memset(classifier_ring, 0, sizeof(classifier_ring));
	memset(classifier_window, 0, sizeof(classifier_window));
	ring_write_index = 0U;
	ring_count = 0U;
	hop_count = 0U;
	sample_id_valid = false;
	last_sample_id16 = 0U;
	unwrapped_sample_id = 0U;
	window_end_sample_id = 0U;
}

static uint32_t unwrap_sample_id(uint16_t sample_id)
{
	if (!sample_id_valid) {
		sample_id_valid = true;
		last_sample_id16 = sample_id;
		unwrapped_sample_id = sample_id;
		return unwrapped_sample_id;
	}

	unwrapped_sample_id += (uint16_t)(sample_id - last_sample_id16);
	last_sample_id16 = sample_id;

	return unwrapped_sample_id;
}

static void classifier_sample_callback(const struct bt_app_imu_sample *samples,
				       size_t sample_count)
{
	static uint32_t dropped_samples;

	for (size_t i = 0U; i < sample_count; i++) {
		int ret = k_msgq_put(&classifier_sample_msgq, &samples[i], K_NO_WAIT);

		if (ret != 0) {
			dropped_samples++;
			if ((dropped_samples % 32U) == 1U) {
				LOG_WRN("Dropped classifier sample (total %u)", dropped_samples);
			}
		}
	}
}

static void classifier_copy_window(void)
{
	uint16_t oldest = ring_write_index;

	for (uint16_t i = 0U; i < CLF_WINDOW_SAMPLES; i++) {
		uint16_t src = (oldest + i) % CLF_WINDOW_SAMPLES;

		memcpy(classifier_window[i], classifier_ring[src], sizeof(classifier_window[i]));
	}
}

static void classifier_process_sample(const struct bt_app_imu_sample *sample)
{
	uint32_t sample_id = unwrap_sample_id(sample->sample_id);
	int decision;

	classifier_ring[ring_write_index][0] = sample->accel_raw[0];
	classifier_ring[ring_write_index][1] = sample->accel_raw[1];
	classifier_ring[ring_write_index][2] = sample->accel_raw[2];
	classifier_ring[ring_write_index][3] = sample->gyro_raw[0];
	classifier_ring[ring_write_index][4] = sample->gyro_raw[1];
	classifier_ring[ring_write_index][5] = sample->gyro_raw[2];

	ring_write_index = (ring_write_index + 1U) % CLF_WINDOW_SAMPLES;
	if (ring_count < CLF_WINDOW_SAMPLES) {
		ring_count++;
	}

	hop_count++;
	window_end_sample_id = sample_id;

	if (ring_count < CLF_WINDOW_SAMPLES || hop_count < CLASSIFIER_HOP_SAMPLES) {
		return;
	}

	hop_count = 0U;
	classifier_copy_window();
	decision = clf_process_window(classifier_window, CLF_WINDOW_SAMPLES);

	if (decision >= 0 && decision < CLF_CLASS_COUNT) {
		bt_app_send_classification(IS_ENABLED(CONFIG_CLASSIFIER_CNN) ?
					   BT_APP_CLASSIFIER_CNN : BT_APP_CLASSIFIER_HDC,
					   (uint8_t)decision,
					   (uint8_t)decision,
					   clf_last_score(),
					   window_end_sample_id);
	}
}

static void classifier_drain_samples(void)
{
	struct bt_app_imu_sample sample;

	while (k_msgq_get(&classifier_sample_msgq, &sample, K_NO_WAIT) == 0) {
		classifier_process_sample(&sample);
	}
}
#else
static void classifier_reset(void)
{
}
#endif

static void handle_control_events(uint32_t events)
{
	LOG_INF("handle_control_events: events=0x%08x", events);

	if ((events & EVENT_ENTER_LOW_POWER) != 0U) {
		LOG_INF("handle_control_events: EVENT_ENTER_LOW_POWER");
		imu_stop_streaming();
		bt_app_mark_stream_stopped();
	}

	if ((events & EVENT_START_STREAMING) != 0U) {
		int ret;

		LOG_INF("handle_control_events: EVENT_START_STREAMING");
		classifier_reset();
		ret = imu_start_streaming();
		LOG_INF("handle_control_events: imu_start_streaming() -> %d", ret);
		if (ret != 0) {
			LOG_ERR("Failed to start IMU stream (err %d)", ret);
			bt_app_mark_stream_stopped();
		}
	}
}

int main(void)
{
	int ret;

	ret = ble_init();
	if (ret != 0) {
		return ret;
	}

	ret = imu_init();
	if (ret != 0) {
		return ret;
	}

	ret = clf_init();
	if (ret != 0) {
		LOG_WRN("Classifier %s init returned %d", clf_name(), ret);
	}

#if !defined(CONFIG_CLASSIFIER_NONE) && !defined(CONFIG_CLASSIFIER_MLC)
	imu_set_sample_callback(classifier_sample_callback);
#endif

	LOG_INF("Firmware ready: classifier=%s", clf_name());

	int64_t main_heartbeat_last_ms = k_uptime_get();

	while (1) {
		uint32_t events = k_event_wait(&sys_events, CONTROL_EVENTS, false, K_NO_WAIT);

		if (events != 0U) {
			k_event_clear(&sys_events, events);
			handle_control_events(events);
		}

		(void)bt_app_stream_should_continue();

		int64_t now_ms = k_uptime_get();

		if ((now_ms - main_heartbeat_last_ms) >= 2000) {
			main_heartbeat_last_ms = now_ms;
			LOG_INF("main loop alive: is_streaming=%d last_events=0x%08x",
				(int)is_streaming, events);
		}

#if !defined(CONFIG_CLASSIFIER_NONE) && !defined(CONFIG_CLASSIFIER_MLC)
		classifier_drain_samples();
#endif
		k_msleep(20);
	}
}
