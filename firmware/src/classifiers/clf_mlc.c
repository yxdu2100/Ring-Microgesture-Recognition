#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/sys/util.h>

#include <lsm6dsv16x_reg.h>

#include "classifier.h"
#include "clf_mlc.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#include "../modules/mlc.h"
#pragma GCC diagnostic pop
#include "../modules/imu_reg.h"

LOG_MODULE_REGISTER(clf_mlc, LOG_LEVEL_INF);

#define FUNC_CFG_ACCESS_EMB_FUNC_REG_ACCESS BIT(7)
#define FUNC_CFG_ACCESS_BANK_MASK BIT(7)
#define MD1_CFG_INT1_EMB_FUNC BIT(1)
#define MLC_INT1_MLC1 BIT(0)
#define MLC_CONF_POLL_INTERVAL_MS 1U
#define MLC_CONF_POLL_TIMEOUT_MS 100U
/* Output codes of the deployed MEMS Studio tree (fold within_user_01,
 * ml/st_trees/within_user_01.h, exported 2026-07-12). MEMS Studio assigns
 * codes in alphabetical label order; they MUST be re-checked against the
 * mlc_results_0_0 table in modules/mlc.h whenever the tree is regenerated.
 */
#define MLC_CODE_INVALID 0xFFU
#define MLC_CODE_FLICKER 0U
#define MLC_CODE_DOUBLEPINCH 1U
#define MLC_CODE_SIDETAP 2U
#define MLC_CODE_NULL 3U
#define MLC_CODE_PINCHHOLD 4U
#define MLC_PROJECT_CLASS_NULL 4U

static atomic_t pending_class = ATOMIC_INIT(-EAGAIN);
static int16_t last_score;
static uint8_t last_reported_raw_code;

static int mlc_select_embedded_bank(bool embedded)
{
	return imu_reg_update(LSM6DSV16X_FUNC_CFG_ACCESS,
			      FUNC_CFG_ACCESS_BANK_MASK,
			      embedded ? FUNC_CFG_ACCESS_EMB_FUNC_REG_ACCESS : 0U);
}

static int mlc_read_mlc1_src(uint8_t *src)
{
	int ret;
	int restore_ret;

	ret = mlc_select_embedded_bank(true);
	if (ret != 0) {
		return ret;
	}

	ret = imu_reg_read(LSM6DSV16X_MLC1_SRC, src);
	restore_ret = mlc_select_embedded_bank(false);

	return ret != 0 ? ret : restore_ret;
}

static int mlc_project_class_from_raw(uint8_t raw_code)
{
	switch (raw_code) {
	case MLC_CODE_SIDETAP:
		return 0;
	case MLC_CODE_DOUBLEPINCH:
		return 1;
	case MLC_CODE_PINCHHOLD:
		return 2;
	case MLC_CODE_FLICKER:
		return 3;
	case MLC_CODE_NULL:
		return MLC_PROJECT_CLASS_NULL;
	default:
		return -EAGAIN;
	}
}

static int mlc_wait_for_register(uint8_t address, uint8_t mask, bool set)
{
	for (uint32_t waited_ms = 0U; waited_ms < MLC_CONF_POLL_TIMEOUT_MS;
	     waited_ms += MLC_CONF_POLL_INTERVAL_MS) {
		uint8_t value = 0U;
		int ret = imu_reg_read(address, &value);

		if (ret != 0) {
			return ret;
		}

		bool matched = set ? ((value & mask) == mask) : ((value & mask) == 0U);

		if (matched) {
			return 0;
		}

		k_msleep(MLC_CONF_POLL_INTERVAL_MS);
	}

	return -ETIMEDOUT;
}

static int mlc_apply_op(const struct mems_conf_op *op)
{
	switch (op->type) {
	case MEMS_CONF_OP_TYPE_READ: {
		uint8_t value = 0U;
		return imu_reg_read(op->address, &value);
	}
	case MEMS_CONF_OP_TYPE_WRITE:
		return imu_reg_write(op->address, op->data);
	case MEMS_CONF_OP_TYPE_DELAY:
		k_msleep(op->data);
		return 0;
	case MEMS_CONF_OP_TYPE_POLL_SET:
		return mlc_wait_for_register(op->address, op->data, true);
	case MEMS_CONF_OP_TYPE_POLL_RESET:
		return mlc_wait_for_register(op->address, op->data, false);
	default:
		return -ENOTSUP;
	}
}

int clf_init(void)
{
	const struct mems_conf_op_list *conf = &mlc_confs[0];

	for (uint32_t i = 0; i < conf->len; i++) {
		int ret = mlc_apply_op(&conf->list[i]);

		if (ret != 0) {
			LOG_ERR("MLC MEMS op %u failed (type %u addr 0x%02x err %d)",
				i, conf->list[i].type, conf->list[i].address, ret);
			return ret;
		}
	}

	last_reported_raw_code = MLC_CODE_INVALID;
	atomic_set(&pending_class, -EAGAIN);
	LOG_INF("MLC classifier initialized from MEMS Studio config (%u ops)", conf->len);
	return 0;
}

int clf_process_window(const int16_t (*win)[CLF_CHANNELS], uint16_t n)
{
	atomic_val_t decision;

	ARG_UNUSED(win);
	ARG_UNUSED(n);

	decision = atomic_get(&pending_class);
	if (decision < 0) {
		return -EAGAIN;
	}

	if (atomic_cas(&pending_class, decision, -EAGAIN)) {
		return (int)decision;
	}

	return -EAGAIN;
}

const char *clf_name(void)
{
	return "mlc";
}

int16_t clf_last_score(void)
{
	return last_score;
}

int clf_mlc_poll_result(uint8_t *class_id, int16_t *score, uint8_t *raw_code)
{
	uint8_t src = 0U;
	int decision;
	int ret = mlc_read_mlc1_src(&src);

	if (ret != 0) {
		LOG_DBG("MLC1_SRC read failed (err %d)", ret);
		return ret;
	}

	decision = mlc_project_class_from_raw(src);
	if (decision < 0) {
		return decision;
	}

	last_score = (int16_t)src;
	atomic_set(&pending_class, (atomic_val_t)decision);

	if (src == last_reported_raw_code && decision == MLC_PROJECT_CLASS_NULL) {
		return -EAGAIN;
	}

	last_reported_raw_code = src;
	if (class_id != NULL) {
		*class_id = (uint8_t)decision;
	}
	if (score != NULL) {
		*score = last_score;
	}
	if (raw_code != NULL) {
		*raw_code = src;
	}

	return 0;
}
