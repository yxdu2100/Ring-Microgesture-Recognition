#ifndef GENERATED_TREE_SW_H
#define GENERATED_TREE_SW_H

#include <stdint.h>

// data_git_hash: 407313a
// cross_session_macro_f1: 0.799685
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
	if (f[18] <= 1146.8811f) {
		if (f[21] <= 19800.2578f) {
			if (f[6] <= 0.0402889289f) {
				if (f[3] <= 1.04138184f) {
					if (f[8] <= 1.39794922f) {
						if (f[5] <= -0.0253458023f) {
							return 2;
						} else {
							return 4;
						}
					} else {
						if (f[0] <= -0.101973534f) {
							return 4;
						} else {
							return 0;
						}
					}
				} else {
					if (f[35] <= 81.3234291f) {
						if (f[24] <= 10.5f) {
							return 2;
						} else {
							return 4;
						}
					} else {
						if (f[36] <= 6754.79004f) {
							return 4;
						} else {
							return 1;
						}
					}
				}
			} else {
				if (f[4] <= 10.5f) {
					return 4;
				} else {
					if (f[15] <= 6.91986084f) {
						if (f[8] <= 1.6529541f) {
							return 4;
						} else {
							return 0;
						}
					} else {
						if (f[26] <= 15072.6807f) {
							return 4;
						} else {
							return 2;
						}
					}
				}
			}
		} else {
			if (f[24] <= 5.5f) {
				return 2;
			} else {
				return 1;
			}
		}
	} else {
		return 3;
	}
}

#endif
