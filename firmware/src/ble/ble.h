#ifndef BLE_H_  
#define BLE_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>

#define BT_APP_IMU_SAMPLE_PAYLOAD_LEN 23U
#define BT_APP_IMU_SAMPLES_PER_PACKET 7U
#define BT_APP_CLASSIFICATION_PAYLOAD_LEN 7U

#define BT_APP_IMU_TS_FLAG_HARDWARE     BIT(0)
#define BT_APP_IMU_TS_FLAG_INTERPOLATED BIT(1)
#define BT_APP_IMU_TS_FLAG_FALLBACK     BIT(2)
#define BT_APP_IMU_TS_FLAG_FIFO_OVERRUN BIT(3)
#define BT_APP_IMU_TS_FLAG_NONMONOTONIC BIT(4)

struct bt_app_imu_sample {
    uint16_t sample_id;
    uint32_t timestamp_us;
    uint32_t timestamp_ticks;
    uint8_t timestamp_flags;
    int16_t accel_raw[3];
    int16_t gyro_raw[3];
};

#define EVENT_START_STREAMING  BIT(0)
#define EVENT_ENTER_LOW_POWER  BIT(1)

extern struct k_event sys_events;
extern volatile bool is_streaming;


int ble_init(void);
void bt_app_send_imu_samples(const struct bt_app_imu_sample *samples, size_t sample_count);
void bt_app_send_classification(uint8_t class_id, int16_t score, uint32_t sample_id);
bool bt_app_is_connected(void);
bool bt_app_stream_should_continue(void);
void bt_app_mark_stream_stopped(void);
void ble_adv_slow(void);
void disconnect_ble(void);
#endif
