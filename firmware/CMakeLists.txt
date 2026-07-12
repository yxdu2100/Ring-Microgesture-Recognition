cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(firmware)

# TFLM uses C++ local statics in the Zephyr samples. Keep that ABI behavior
# explicit when the CNN classifier is selected.
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

target_include_directories(app PRIVATE
    src
    src/ble
    src/modules
    src/classifiers
    src/classifiers/generated
)

set_source_files_properties(src/classifiers/clf_cnn.c PROPERTIES LANGUAGE CXX)

target_sources(app PRIVATE
    src/main.c
    src/benchmark.c
    src/modules/imu.c
)

target_sources_ifdef(CONFIG_BT app PRIVATE src/ble/ble.c)

target_sources_ifdef(CONFIG_CLASSIFIER_NONE app PRIVATE src/classifiers/clf_none.c)
target_sources_ifdef(CONFIG_CLASSIFIER_MLC  app PRIVATE src/classifiers/clf_mlc.c)
target_sources_ifdef(CONFIG_CLASSIFIER_CNN  app PRIVATE src/classifiers/clf_cnn.c)
target_sources_ifdef(CONFIG_CLASSIFIER_HDC  app PRIVATE src/classifiers/clf_hdc.c)
