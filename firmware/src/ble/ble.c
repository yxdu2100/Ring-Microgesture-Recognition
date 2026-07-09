#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/bluetooth/addr.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/sys/util.h>

#include "ble.h"
#include "../modules/imu.h"

LOG_MODULE_REGISTER(ble_module, LOG_LEVEL_INF);

#define ADVERTISING_INTERVAL 12800
#define STREAM_LEASE_TIMEOUT_MS 3000
#define ATTR_IDX_CMD_DATA_VAL 2
#define ATTR_IDX_IMU_DATA_VAL 7

#define BT_UUID_RING_SERVICE_VAL \
	BT_UUID_128_ENCODE(0x12345678, 0x9abc, 0x11ee, 0xbe56, 0x0242ac120002)
#define BT_UUID_RING_SERVICE BT_UUID_DECLARE_128(BT_UUID_RING_SERVICE_VAL)

#define BT_UUID_CMD_DATA_VAL \
	BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcde01)
#define BT_UUID_CMD_DATA BT_UUID_DECLARE_128(BT_UUID_CMD_DATA_VAL)

#define BT_UUID_IMU_MODE_VAL \
	BT_UUID_128_ENCODE(0x1234567B, 0x9abc, 0x11ee, 0xbe56, 0x0242ac120002)
#define BT_UUID_IMU_MODE BT_UUID_DECLARE_128(BT_UUID_IMU_MODE_VAL)

#define BT_UUID_IMU_DATA_VAL \
	BT_UUID_128_ENCODE(0x1234567D, 0x9abc, 0x11ee, 0xbe56, 0x0242ac120002)
#define BT_UUID_IMU_DATA BT_UUID_DECLARE_128(BT_UUID_IMU_DATA_VAL)

static struct bt_conn *current_conn;
static bool cmd_notify_enabled;
static bool imu_notify_enabled;
static uint8_t imu_dummy_buf[1];
static uint8_t current_imu_mode_val;
static int64_t last_stream_lease_ms;
static bool recovery_in_progress;

K_EVENT_DEFINE(sys_events);
volatile bool is_streaming;

const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME,
		sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

const struct bt_data sd[] = {
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_RING_SERVICE_VAL),
};

static const struct bt_le_adv_param *adv_param_slow =
	BT_LE_ADV_PARAM(BT_LE_ADV_OPT_CONNECTABLE,
			ADVERTISING_INTERVAL,
			ADVERTISING_INTERVAL,
			NULL);

static void stop_streaming_from_ble(void);

void mtu_updated(struct bt_conn *conn, uint16_t tx, uint16_t rx)
{
	ARG_UNUSED(conn);
	LOG_INF("Updated MTU: TX: %d RX: %d bytes", tx, rx);
}

static struct bt_gatt_cb gatt_callbacks = {
	.att_mtu_updated = mtu_updated,
};

static ssize_t read_u8_attr(struct bt_conn *conn, const struct bt_gatt_attr *attr,
			    void *buf, uint16_t len, uint16_t offset)
{
	const uint8_t *value = attr->user_data;

	return bt_gatt_attr_read(conn, attr, buf, len, offset, value, sizeof(*value));
}

static bool stream_transport_ready(void)
{
	if (current_conn == NULL) {
		return false;
	}

	if (IS_ENABLED(CONFIG_CLASSIFIER_NONE) ||
	    IS_ENABLED(CONFIG_CLASSIFIER_DEBUG_STREAM)) {
		return imu_notify_enabled;
	}

	return cmd_notify_enabled;
}

static void refresh_stream_lease(void)
{
	last_stream_lease_ms = k_uptime_get();
}

static bool stream_lease_expired(void)
{
	if (!is_streaming) {
		return false;
	}

	return (k_uptime_get() - last_stream_lease_ms) > STREAM_LEASE_TIMEOUT_MS;
}

static void recover_stream_transport(const char *reason)
{
	if (recovery_in_progress) {
		return;
	}

	recovery_in_progress = true;
	LOG_WRN("Recovering BLE stream transport: %s", reason);
	stop_streaming_from_ble();

	if (current_conn) {
		int err = bt_conn_disconnect(current_conn, BT_HCI_ERR_REMOTE_USER_TERM_CONN);

		if (err != 0 && err != -ENOTCONN) {
			LOG_ERR("Failed to disconnect during recovery (err %d)", err);
		}
	}

	recovery_in_progress = false;
}

ssize_t on_phone_command_received(struct bt_conn *conn, const struct bt_gatt_attr *attr,
				  const void *buf, uint16_t len, uint16_t offset,
				  uint8_t flags)
{
	uint8_t command;

	ARG_UNUSED(conn);
	ARG_UNUSED(attr);
	ARG_UNUSED(flags);

	if (offset != 0U || len != 1U) {
		return BT_GATT_ERR(BT_ATT_ERR_INVALID_ATTRIBUTE_LEN);
	}

	command = *((const uint8_t *)buf);
	LOG_INF("Received command: %d", command);

	if (command == 1U) {
		if (!is_streaming) {
			LOG_INF("CMD: start");
			is_streaming = true;
			refresh_stream_lease();
			k_event_post(&sys_events, EVENT_START_STREAMING);
		}
	} else if (command == 0U) {
		if (is_streaming) {
			LOG_INF("CMD: stop");
			is_streaming = false;
			k_event_post(&sys_events, EVENT_ENTER_LOW_POWER);
		}
	} else if (command == 3U) {
		refresh_stream_lease();
		LOG_DBG("CMD: stream keepalive");
	} else {
		LOG_WRN("Ignoring unknown command: %u", command);
	}

	return len;
}

ssize_t on_imu_mode_received(struct bt_conn *conn, const struct bt_gatt_attr *attr,
			     const void *buf, uint16_t len, uint16_t offset,
			     uint8_t flags)
{
	uint8_t requested_mode;

	ARG_UNUSED(conn);
	ARG_UNUSED(attr);
	ARG_UNUSED(flags);

	if (offset != 0U || len != 1U) {
		LOG_ERR("Invalid IMU mode payload length: %d", len);
		return BT_GATT_ERR(BT_ATT_ERR_INVALID_ATTRIBUTE_LEN);
	}

	requested_mode = *((const uint8_t *)buf);
	LOG_INF("BLE received IMU mode switch request: %d", requested_mode);
	current_imu_mode_val = requested_mode;
	(void)imu_set_trigger_mode(current_imu_mode_val);

	return len;
}

static void cmd_ccc_cfg_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
	ARG_UNUSED(attr);
	cmd_notify_enabled = (value == BT_GATT_CCC_NOTIFY);
	LOG_INF("CMD notify %s", cmd_notify_enabled ? "enabled" : "disabled");
}

static void imu_ccc_cfg_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
	ARG_UNUSED(attr);
	imu_notify_enabled = (value == BT_GATT_CCC_NOTIFY);
	LOG_INF("IMU notify %s", imu_notify_enabled ? "enabled" : "disabled");
}

BT_GATT_SERVICE_DEFINE(ring_svc,
	BT_GATT_PRIMARY_SERVICE(BT_UUID_RING_SERVICE),

	BT_GATT_CHARACTERISTIC(BT_UUID_CMD_DATA,
		BT_GATT_CHRC_WRITE | BT_GATT_CHRC_WRITE_WITHOUT_RESP | BT_GATT_CHRC_NOTIFY,
		BT_GATT_PERM_WRITE,
		NULL, on_phone_command_received, NULL),

	BT_GATT_CCC(cmd_ccc_cfg_changed,
		BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),

	BT_GATT_CHARACTERISTIC(BT_UUID_IMU_MODE,
		BT_GATT_CHRC_WRITE | BT_GATT_CHRC_WRITE_WITHOUT_RESP | BT_GATT_CHRC_READ,
		BT_GATT_PERM_WRITE | BT_GATT_PERM_READ,
		read_u8_attr,
		on_imu_mode_received,
		&current_imu_mode_val),

	BT_GATT_CHARACTERISTIC(BT_UUID_IMU_DATA,
		BT_GATT_CHRC_NOTIFY,
		0,
		NULL,
		NULL,
		imu_dummy_buf),

	BT_GATT_CCC(imu_ccc_cfg_changed,
		BT_GATT_PERM_READ | BT_GATT_PERM_WRITE)
);

static void stop_streaming_from_ble(void)
{
	if (is_streaming) {
		is_streaming = false;
		k_event_post(&sys_events, EVENT_ENTER_LOW_POWER);
	}
}

void update_conn_params(struct bt_conn *conn)
{
	struct bt_le_conn_param *param = BT_LE_CONN_PARAM(12, 12, 0, 400);
	int err = bt_conn_le_param_update(conn, param);

	if (err) {
		LOG_ERR("Conn param update failed (err %d)", err);
	} else {
		LOG_INF("Requested 15ms connection interval");
	}
}

void update_phy(struct bt_conn *conn)
{
	const struct bt_conn_le_phy_param param = {
		.options = BT_CONN_LE_PHY_OPT_NONE,
		.pref_tx_phy = BT_GAP_LE_PHY_2M,
		.pref_rx_phy = BT_GAP_LE_PHY_2M,
	};
	int err = bt_conn_le_phy_update(conn, &param);

	if (err) {
		LOG_WRN("PHY update failed (err %d)", err);
	} else {
		LOG_INF("Requested 2M PHY speed");
	}
}

static void connected(struct bt_conn *conn, uint8_t err)
{
	if (err != 0U) {
		LOG_ERR("Connection failed (err %u)", err);
		return;
	}

	current_conn = bt_conn_ref(conn);
	LOG_INF("Connected");
	refresh_stream_lease();
	update_conn_params(conn);
	update_phy(conn);
}

#define ADV_RESTART_RETRY_DELAY_MS 50
#define ADV_RESTART_MAX_ATTEMPTS 20

static int adv_restart_attempts;

static void restart_advertising_work_handler(struct k_work *work);

static K_WORK_DELAYABLE_DEFINE(restart_advertising_work, restart_advertising_work_handler);

static void restart_advertising_work_handler(struct k_work *work)
{
	ARG_UNUSED(work);

	int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1,
				  ad, ARRAY_SIZE(ad),
				  sd, ARRAY_SIZE(sd));

	if (err) {
		adv_restart_attempts++;
		LOG_ERR("Advertising failed to restart (err %d), attempt %d",
			err, adv_restart_attempts);
		if (adv_restart_attempts < ADV_RESTART_MAX_ATTEMPTS) {
			k_work_schedule(&restart_advertising_work,
					K_MSEC(ADV_RESTART_RETRY_DELAY_MS));
		}
		return;
	}

	adv_restart_attempts = 0;
	LOG_INF("Advertising restarted");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	ARG_UNUSED(conn);

	LOG_INF("Disconnected (reason 0x%02x)", reason);

	if (current_conn) {
		bt_conn_unref(current_conn);
		current_conn = NULL;
	}

	stop_streaming_from_ble();
	last_stream_lease_ms = 0;
	cmd_notify_enabled = false;
	imu_notify_enabled = false;

	adv_restart_attempts = 0;
	k_work_schedule(&restart_advertising_work, K_NO_WAIT);
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
};

static void log_ble_identity_diagnostics(void)
{
	bt_addr_le_t identities[CONFIG_BT_ID_MAX];
	size_t id_count = ARRAY_SIZE(identities);

	LOG_INF("Compiled BT_DEVICE_NAME=\"%s\" (%u bytes)",
		CONFIG_BT_DEVICE_NAME, (unsigned)(sizeof(CONFIG_BT_DEVICE_NAME) - 1));
	LOG_INF("Compiled service UUID (ad/sd payload) - see BT_UUID_RING_SERVICE_VAL in ble.c");

	bt_id_get(identities, &id_count);
	LOG_INF("Bluetooth identity count in use: %u (CONFIG_BT_ID_MAX=%d)",
		(unsigned)id_count, CONFIG_BT_ID_MAX);

	for (size_t i = 0; i < id_count; i++) {
		char addr_str[BT_ADDR_LE_STR_LEN];

		bt_addr_le_to_str(&identities[i], addr_str, sizeof(addr_str));
		LOG_INF("  identity[%u] address: %s", (unsigned)i, addr_str);
	}
}

int ble_init(void)
{
	int err;

	bt_gatt_cb_register(&gatt_callbacks);

	err = bt_enable(NULL);
	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}

	LOG_INF("Bluetooth initialized");
	log_ble_identity_diagnostics();

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1,
			      ad, ARRAY_SIZE(ad),
			      sd, ARRAY_SIZE(sd));
	if (err) {
		LOG_ERR("Advertising failed to start (err %d)", err);
		return err;
	}

	LOG_INF("Advertising started on identity %d", BT_ID_DEFAULT);
	return 0;
}

void bt_app_send_imu_samples(const struct bt_app_imu_sample *samples, size_t sample_count)
{
	uint8_t payload[BT_APP_IMU_SAMPLE_PAYLOAD_LEN * BT_APP_IMU_SAMPLES_PER_PACKET];
	static uint32_t dropped_imu_batches;

	if (!samples || sample_count == 0 || !current_conn || !imu_notify_enabled || !is_streaming) {
		return;
	}

	while (sample_count > 0) {
		size_t samples_in_packet = MIN(sample_count, (size_t)BT_APP_IMU_SAMPLES_PER_PACKET);

		for (size_t i = 0; i < samples_in_packet; i++) {
			uint8_t *sample_payload = &payload[i * BT_APP_IMU_SAMPLE_PAYLOAD_LEN];
			const struct bt_app_imu_sample *sample = &samples[i];

			sys_put_le16(sample->sample_id, &sample_payload[0]);
			sys_put_le32(sample->timestamp_us, &sample_payload[2]);
			sys_put_le32(sample->timestamp_ticks, &sample_payload[6]);
			sample_payload[10] = sample->timestamp_flags;
			sys_put_le16((uint16_t)sample->accel_raw[0], &sample_payload[11]);
			sys_put_le16((uint16_t)sample->accel_raw[1], &sample_payload[13]);
			sys_put_le16((uint16_t)sample->accel_raw[2], &sample_payload[15]);
			sys_put_le16((uint16_t)sample->gyro_raw[0], &sample_payload[17]);
			sys_put_le16((uint16_t)sample->gyro_raw[1], &sample_payload[19]);
			sys_put_le16((uint16_t)sample->gyro_raw[2], &sample_payload[21]);
		}

		int err = bt_gatt_notify(current_conn,
					 &ring_svc.attrs[ATTR_IDX_IMU_DATA_VAL],
					 payload,
					 samples_in_packet * BT_APP_IMU_SAMPLE_PAYLOAD_LEN);
		if (err != 0) {
			dropped_imu_batches++;
			if ((dropped_imu_batches % 32U) == 1U) {
				LOG_WRN("Dropped IMU BLE batch (err %d, total %u)",
					err, dropped_imu_batches);
			}
		} else {
			refresh_stream_lease();
		}

		samples += samples_in_packet;
		sample_count -= samples_in_packet;
	}
}

void bt_app_send_classification(uint8_t class_id, int16_t score, uint32_t sample_id)
{
	uint8_t payload[BT_APP_CLASSIFICATION_PAYLOAD_LEN];
	int err;

	if (!current_conn || !cmd_notify_enabled || !is_streaming) {
		return;
	}

	payload[0] = class_id;
	sys_put_le16((uint16_t)score, &payload[1]);
	sys_put_le32(sample_id, &payload[3]);

	err = bt_gatt_notify(current_conn,
			     &ring_svc.attrs[ATTR_IDX_CMD_DATA_VAL],
			     payload,
			     sizeof(payload));
	if (err != 0) {
		LOG_WRN("Dropped classification result (err %d)", err);
	} else {
		refresh_stream_lease();
	}
}

bool bt_app_is_connected(void)
{
	return current_conn != NULL;
}

void bt_app_mark_stream_stopped(void)
{
	uint32_t pending = k_event_wait(&sys_events,
					EVENT_START_STREAMING,
					false,
					K_NO_WAIT);

	if (pending & EVENT_START_STREAMING) {
		return;
	}

	is_streaming = false;
}

bool bt_app_stream_should_continue(void)
{
	if (!is_streaming) {
		return false;
	}

	if (!current_conn) {
		LOG_WRN("Stream halted: no active connection");
		is_streaming = false;
		return false;
	}

	if (!stream_transport_ready()) {
		LOG_WRN("Stream halted: transport not ready");
		is_streaming = false;
		return false;
	}

	if (stream_lease_expired()) {
		LOG_WRN("Stream lease expired while idle");
		recover_stream_transport("heartbeat timeout");
		return false;
	}

	return true;
}

void ble_adv_slow(void)
{
	int err = bt_le_adv_start(adv_param_slow, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));

	if (err) {
		LOG_ERR("Advertising failed to start (err %d)", err);
		return;
	}

	LOG_INF("Low power advertising started");
}

void disconnect_ble(void)
{
	if (current_conn) {
		(void)bt_conn_disconnect(current_conn, BT_HCI_ERR_REMOTE_USER_TERM_CONN);
	}
}
