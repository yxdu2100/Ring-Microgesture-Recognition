#ifndef GENERATED_TREE_SW_H
#define GENERATED_TREE_SW_H

#include <stdint.h>

// data_git_hash: d687358
// cross_session_macro_f1: 0.783424
#define TREE_SW_FEATURE_COUNT 40U
#define TREE_SW_MAX_DEPTH 6U

static const char *const tree_sw_feature_names[TREE_SW_FEATURE_COUNT] = {
	"ax_mean",
	"ax_variance",
	"ax_energy",
	"ax_peak_to_peak",
	"ax_zero_crossings",
	"ay_mean",
	"ay_variance",
	"ay_energy",
	"ay_peak_to_peak",
	"ay_zero_crossings",
	"az_mean",
	"az_variance",
	"az_energy",
	"az_peak_to_peak",
	"az_zero_crossings",
	"gx_mean",
	"gx_variance",
	"gx_energy",
	"gx_peak_to_peak",
	"gx_zero_crossings",
	"gy_mean",
	"gy_variance",
	"gy_energy",
	"gy_peak_to_peak",
	"gy_zero_crossings",
	"gz_mean",
	"gz_variance",
	"gz_energy",
	"gz_peak_to_peak",
	"gz_zero_crossings",
	"accel_norm_mean",
	"accel_norm_variance",
	"accel_norm_energy",
	"accel_norm_peak_to_peak",
	"accel_norm_zero_crossings",
	"gyro_norm_mean",
	"gyro_norm_variance",
	"gyro_norm_energy",
	"gyro_norm_peak_to_peak",
	"gyro_norm_zero_crossings",
};

static inline int tree_sw_predict(const float f[TREE_SW_FEATURE_COUNT])
{
	if (f[18] <= 1247.10083f) {
		if (f[23] <= 806.152344f) {
			if (f[38] <= 275.3116f) {
				if (f[9] <= 1.5f) {
					return 4;
				} else {
					return 4;
				}
			} else {
				if (f[30] <= 1.04131758f) {
					if (f[11] <= 0.0513880197f) {
						if (f[16] <= 3857.87036f) {
							return 2;
						} else {
							return 4;
						}
					} else {
						return 4;
					}
				} else {
					if (f[2] <= 0.0542657673f) {
						if (f[11] <= 0.103777312f) {
							return 0;
						} else {
							return 4;
						}
					} else {
						if (f[35] <= 69.4203644f) {
							return 2;
						} else {
							return 4;
						}
					}
				}
			}
		} else {
			if (f[34] <= 11.5f) {
				return 1;
			} else {
				return 1;
			}
		}
	} else {
		if (f[17] <= 36026.627f) {
			return 3;
		} else {
			return 3;
		}
	}
}

#endif
