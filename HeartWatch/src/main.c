// Include libraries here
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "esp_log.h"

// Macros for GPIO
#define USER_LED GPIO_NUM_21
#define I2C_SCL GPIO_NUM_6
#define I2C_SDA GPIO_NUM_5

// Configs
#define MPU_DATA_LENGTH 14
#define MPU_REG_WHO_AM_I 0x75
#define MPU_REG_ACCEL_XOUT_H 0x3B

static const char *TAG = "mpu";

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

    // MPU read buffer
    uint8_t mpu_data[MPU_DATA_LENGTH];
    uint8_t register_address;
    esp_err_t result;

    // Read WHO_AM_I to verify that the MPU is responding at address 0x68.
    register_address = MPU_REG_WHO_AM_I;
    result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
                                         mpu_data, 1, -1);
    ESP_ERROR_CHECK(result);
    ESP_LOGI(TAG, "WHO_AM_I: 0x%02X", mpu_data[0]);

    // Main polling loop
	while (1) {
        
        // MPU reading output to error log
        // ESP_LOGI(TAG, "WHO_AM_I: 0x%02X", mpu_data[0]);
		// register_address = MPU_REG_ACCEL_XOUT_H;
		// result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
		//                                      mpu_data, sizeof(mpu_data), -1);
		// ESP_ERROR_CHECK(result);
		// ESP_LOG_BUFFER_HEXDUMP(TAG, mpu_data, sizeof(mpu_data), ESP_LOG_INFO);

        // Heartbeat LED
		gpio_set_level(USER_LED, 0);
		vTaskDelay(pdMS_TO_TICKS(250));
		gpio_set_level(USER_LED, 1);
		vTaskDelay(pdMS_TO_TICKS(250));
	}
}

