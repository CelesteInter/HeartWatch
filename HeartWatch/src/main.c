// Include libraries here
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "esp_timer.h"

// Custom headers
#include "max30102.h"

// Macros for GPIO
#define USER_LED GPIO_NUM_21
#define I2C_SCL GPIO_NUM_6
#define I2C_SDA GPIO_NUM_5

// Configs

/// Data length
#define MPU_DATA_LENGTH 14

// Addresses
#define WHO_AM_I 0x75
#define MPU_REG_ACCEL_XOUT_H 0x3B

// Tags
static const char *TAG = "mpu";
static const char *HEART_RATE_TAG = "heartrate";


// Main application loop
void app_main(void)
{
    // Config

    // Heartbeat LED config
	gpio_config_t led_config = {
		.pin_bit_mask = 1ULL << USER_LED,
		.mode = GPIO_MODE_OUTPUT,
		.pull_up_en = GPIO_PULLUP_DISABLE,
		.pull_down_en = GPIO_PULLDOWN_DISABLE,
		.intr_type = GPIO_INTR_DISABLE,
	};
	gpio_config(&led_config);

    // I2C master configuration
    i2c_master_bus_config_t i2c_mst_config = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .i2c_port = I2C_NUM_1,
        .scl_io_num = I2C_SCL,
        .sda_io_num = I2C_SDA,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    // bus handler (to be used for calls after setup)
    i2c_master_bus_handle_t bus_handle;
    // Error checking & bus setup (verifies I2C bus setup correctly)
    ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_mst_config, &bus_handle));

    // MPU I2C configuration
    i2c_device_config_t i2c_mpu_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = 0x68,
        .scl_speed_hz = 100000,
    };
    // bus handler
    i2c_master_dev_handle_t mpu_handle;
    // Error checking & bus setup (verify config & setup)
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &i2c_mpu_config, &mpu_handle));

    // Delay for terminal to catchup for debug
    vTaskDelay(500);

    // MPU read buffer
    uint8_t mpu_data[MPU_DATA_LENGTH];
    uint8_t register_address;
    esp_err_t result;

    // Read WHO_AM_I to verify that the MPU is responding at address 0x68.
    register_address = WHO_AM_I;
    result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
                                         mpu_data, 1, -1);
    ESP_ERROR_CHECK(result);
    ESP_LOGI(TAG, "WHO_AM_I: 0x%02X", mpu_data[0]);

    // MAX30102 I2C configuration
    i2c_device_config_t i2c_heartrate_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MAX30102_ADDRESS,
        .scl_speed_hz = 100000,
    };
    // bus handler
    i2c_master_dev_handle_t heartrate_handle;
    // Error checking & bus setup
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &i2c_heartrate_config, &heartrate_handle));

    ESP_ERROR_CHECK(max30102_init(heartrate_handle));
    ESP_LOGI(HEART_RATE_TAG, "MAX30102 initialized");

    // variables to hold MAX30102 values
    uint32_t ir_value;
    uint32_t ir_baseline = 0;
    uint32_t signal_peak = 0;
    int64_t last_beat_ms = 0;


    // Main polling loop
	while (1) {

        // Read Heartrate BPM
        for (int sample_count = 0; sample_count < 8; sample_count++) {
            if (!max30102_read_ir(heartrate_handle, &ir_value)) {
                break;
            }

            if (ir_baseline == 0) {
                ir_baseline = ir_value;
            }

            // Track the slowly changing DC level and detect a pulse above it.
            // baseline value for ambient light
            ir_baseline = (ir_baseline * 31 + ir_value) / 32;
            // Remove slowly changing DC component from IR reading
            uint32_t signal = ir_value > ir_baseline ? ir_value - ir_baseline : 0;
            // update previous peak from signal
            uint32_t previous_peak = signal_peak;
            // update current signal peak (new peak is 7/8ths old peak, 1/8th new signal)
            signal_peak = (signal_peak * 7 + signal) / 8;
            // Get the current time on the timer (ms)
            int64_t now_ms = esp_timer_get_time() / 1000;

            /**
             * Heartbeat recognition
             * ir_value > 5000 : assumes finger is present, otherwise ambient pulse
             * signal > 500 : pulse must be at least 500 counts above baseline
             * signal > previous_peak * 3/2 : signal must be 50% larger than recent smoothed level
             * now_ms - last_beat_ms > 300 : theoretical maximum limit for BPM of 200 
             *          -> 60000 ms in a minute / 300 ms = 200 BPM
             */
            if (ir_value > 50000 && signal > 500 && signal > previous_peak * 3 / 2 &&
                now_ms - last_beat_ms > 300) {
                // Ensure first detected pulse doesn't provide a data entry
                if (last_beat_ms != 0) {
                    // calculate bpm (60000 ms in a minute / change in time)
                    uint32_t bpm = 60000 / (now_ms - last_beat_ms);
                    // If the BPM is reasonable, print it to log
                    if (bpm >= 40 && bpm <= 220) {
                        ESP_LOGE(HEART_RATE_TAG, "Heart rate: %lu BPM", (unsigned long)bpm);
                    }
                }
                // update last time for beat
                last_beat_ms = now_ms;
            }
        }

        // MPU reading output to error log
        // ESP_LOGI(TAG, "WHO_AM_I: 0x%02X", mpu_data[0]);
		// register_address = MPU_REG_ACCEL_XOUT_H;
		// result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
		//                                      mpu_data, sizeof(mpu_data), -1);
		// ESP_ERROR_CHECK(result);
		// ESP_LOG_BUFFER_HEXDUMP(TAG, mpu_data, sizeof(mpu_data), ESP_LOG_INFO);
        

        // Heartbeat LED
		// gpio_set_level(USER_LED, 0);
		// vTaskDelay(pdMS_TO_TICKS(250));
		// gpio_set_level(USER_LED, 1);
		// vTaskDelay(pdMS_TO_TICKS(250));
	}
}

