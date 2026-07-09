#ifndef IMU_H
#define IMU_H

#include <stdint.h>
#include <stddef.h>

#include "../ble/ble.h"

typedef void (*imu_sample_callback_t)(const struct bt_app_imu_sample *samples,
				      size_t sample_count);

int imu_init(void);
int imu_set_trigger_mode(uint8_t mode);
int imu_start_streaming(void);
void imu_stop_streaming(void);
void imu_set_sample_callback(imu_sample_callback_t callback);

#endif
