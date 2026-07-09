#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/sys/util.h>

#include <lsm6dsv16x_reg.h>

#include "classifier.h"
#include "clf_mlc.h"
#include "generated/mlc_ucf.h"
#include "../modules/imu_reg.h"

LOG_MODULE_REGISTER(clf_mlc, LOG_LEVEL_INF);

#define FUNC_CFG_ACCESS_EMB_FUNC_REG_ACCESS BIT(7)
#define FUNC_CFG_ACCESS_BANK_MASK BIT(7)
#define MD1_CFG_INT1_EMB_FUNC BIT(1)
#define MLC_INT1_MLC1 BIT(0)

static atomic_t pending_class = ATOMIC_INIT(-EAGAIN);
static int16_t last_score;

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

int clf_init(void)
{
	int ret = 0;

	for (size_t i = 0; i < CLF_MLC_UCF_LEN; i++) {
		ret = imu_reg_write(clf_mlc_ucf[i].address, clf_mlc_ucf[i].data);
		if (ret != 0) {
			LOG_ERR("MLC UCF write failed at line %u (err %d)", i, ret);
			return ret;
		}
	}

	ret = mlc_select_embedded_bank(true);
	if (ret != 0) {
		return ret;
	}

	{
		int update_ret = imu_reg_update(LSM6DSV16X_MLC_INT1, MLC_INT1_MLC1,
						MLC_INT1_MLC1);
		int restore_ret = mlc_select_embedded_bank(false);

		ret = update_ret != 0 ? update_ret : restore_ret;
		if (ret != 0) {
			return ret;
		}
	}

	ret = imu_reg_update(LSM6DSV16X_MD1_CFG, MD1_CFG_INT1_EMB_FUNC, MD1_CFG_INT1_EMB_FUNC);
	if (ret != 0) {
		return ret;
	}

	LOG_INF("MLC classifier initialized (%u UCF lines)", CLF_MLC_UCF_LEN);
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

void clf_mlc_irq_handler(void)
{
	uint8_t src = 0U;
	int ret = mlc_read_mlc1_src(&src);

	if (ret != 0) {
		LOG_DBG("MLC1_SRC read failed (err %d)", ret);
		return;
	}

	if (src >= CLF_CLASS_COUNT) {
		return;
	}

	last_score = (int16_t)src;
	atomic_set(&pending_class, (atomic_val_t)src);
}
