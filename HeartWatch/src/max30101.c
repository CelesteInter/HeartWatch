#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "max30101.h"

static const char *TAG = "max30101";
static i2c_master_dev_handle_t device_handle;
static bool initialized;
static bool sleeping;

#define MAX30101_MODE_SHUTDOWN (1U << 7)

// NOTE - Change the -1 on i2c_master_transmit to enable timeout for optimization and time constraints
static esp_err_t max30101_write(i2c_master_dev_handle_t handle,
                                uint8_t reg, uint8_t value)
{
    uint8_t data[] = {reg, value};
    return i2c_master_transmit(handle, data, sizeof(data), -1);
}

static esp_err_t max30101_read(i2c_master_dev_handle_t handle,
                               uint8_t reg, uint8_t *data, size_t length)
{
    return i2c_master_transmit_receive(handle, &reg, 1, data, length, -1);
}

esp_err_t max30101_init(i2c_master_bus_handle_t bus_handle)
{
    if (initialized) {
        return ESP_OK;
    }
    if (bus_handle == NULL) {
        ESP_LOGE(TAG, "Cannot initialize MAX30101 without an I2C bus");
        return ESP_ERR_INVALID_ARG;
    }

    i2c_device_config_t device_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MAX30101_ADDRESS,
        .scl_speed_hz = 100000,
    };
    esp_err_t result = i2c_master_bus_add_device(
        bus_handle, &device_config, &device_handle);
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add MAX30101 device");
        return result;
    }

    uint8_t part_id;
    result = max30101_read(device_handle, MAX30101_REG_PART_ID, &part_id, 1);
    if (result != ESP_OK) {
        return result;
    }
    if (part_id != MAX30101_PART_ID) {
        ESP_LOGE(TAG, "Unexpected MAX30101 part ID: 0x%02X", part_id);
        return ESP_ERR_NOT_FOUND;
    }

    ESP_ERROR_CHECK(max30101_write(device_handle, MAX30101_REG_MODE_CONFIG, 0x40)); // RESET mode
    vTaskDelay(pdMS_TO_TICKS(10)); // Delay to allow reset of MAX30101

    // These pointer resets are optional - RST write to mode configuration should set these to POR value (00)
    ESP_ERROR_CHECK(max30101_write(device_handle, MAX30101_REG_FIFO_WR_PTR, 0x00)); // Set write pointer to 0 (optional)
    ESP_ERROR_CHECK(max30101_write(device_handle, MAX30101_REG_OVF_COUNTER, 0x00)); // Reset overflow counter
    ESP_ERROR_CHECK(max30101_write(device_handle, MAX30101_REG_FIFO_RD_PTR, 0x00)); // Reset read pointer
    ESP_ERROR_CHECK(max30101_write(device_handle, MAX30101_REG_SPO2_CONFIG, 0x27)); // Set SPO2 Configuration
    ESP_ERROR_CHECK(max30101_write(device_handle, MAX30101_REG_LED3_PA, 0x24)); // Pulse 7.2mA
    ESP_ERROR_CHECK(max30101_write(device_handle, MAX30101_REG_MULTI_LED_CTRL1, 0x03)); // Green in slot 1
    ESP_ERROR_CHECK(max30101_write(device_handle, MAX30101_REG_MODE_CONFIG, 0x07)); // Set to multi-LED mode
    initialized = true;
    sleeping = false;
    return ESP_OK;
}

esp_err_t max30101_sleep(void)
{
    if (!initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    if (sleeping) {
        return ESP_OK;
    }

    uint8_t mode_config;
    esp_err_t result = max30101_read(
        device_handle, MAX30101_REG_MODE_CONFIG, &mode_config, 1);
    if (result != ESP_OK) {
        return result;
    }

    result = max30101_write(
        device_handle, MAX30101_REG_MODE_CONFIG,
        mode_config | MAX30101_MODE_SHUTDOWN);
    if (result == ESP_OK) {
        sleeping = true;
    }
    return result;
}

esp_err_t max30101_wakeup(void)
{
    if (!initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    esp_err_t result = max30101_write(
        device_handle, MAX30101_REG_MODE_CONFIG, 0x07);
    if (result != ESP_OK) {
        return result;
    }

    result = max30101_write(device_handle, MAX30101_REG_SPO2_CONFIG, 0x27);
    if (result != ESP_OK) {
        return result;
    }
    result = max30101_write(device_handle, MAX30101_REG_LED3_PA, 0x24);
    if (result != ESP_OK) {
        return result;
    }
    result = max30101_write(device_handle, MAX30101_REG_MULTI_LED_CTRL1, 0x03);
    if (result != ESP_OK) {
        return result;
    }

    result = max30101_write(device_handle, MAX30101_REG_FIFO_WR_PTR, 0x00);
    if (result == ESP_OK) {
        result = max30101_write(device_handle, MAX30101_REG_OVF_COUNTER, 0x00);
    }
    if (result == ESP_OK) {
        result = max30101_write(device_handle, MAX30101_REG_FIFO_RD_PTR, 0x00);
    }
    if (result == ESP_OK) {
        sleeping = false;
        vTaskDelay(pdMS_TO_TICKS(50));
    }
    return result;
}

bool max30101_read_green(uint32_t *green_value)
{
    uint8_t fifo_data[3];

    if (green_value == NULL) {
        ESP_LOGI(TAG, "green was null");
        return false;
    }

    if (!initialized || sleeping) {
        return false;
    }

    if (max30101_read(device_handle, MAX30101_REG_FIFO_DATA, fifo_data, sizeof(fifo_data)) != ESP_OK) {
        ESP_LOGI(TAG, "read failed!");
        return false;
    }

    *green_value = ((uint32_t)(fifo_data[0] & 0x03) << 16) |
                   ((uint32_t)fifo_data[1] << 8) | fifo_data[2];
    return true;
}