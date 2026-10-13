#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "max30101.h"

static const char *TAG = "max30101";

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

esp_err_t max30101_init(i2c_master_dev_handle_t handle)
{
    uint8_t part_id;
    esp_err_t result = max30101_read(handle, MAX30101_REG_PART_ID, &part_id, 1);
    if (result != ESP_OK) {
        return result;
    }
    if (part_id != MAX30101_PART_ID) {
        ESP_LOGE(TAG, "Unexpected MAX30101 part ID: 0x%02X", part_id);
        return ESP_ERR_NOT_FOUND;
    }

    ESP_ERROR_CHECK(max30101_write(handle, MAX30101_REG_MODE_CONFIG, 0x40)); // RESET mode
    vTaskDelay(pdMS_TO_TICKS(10)); // Delay to allow reset of MAX30101

    // These pointer resets are optional - RST write to mode configuration should set these to POR value (00)
    ESP_ERROR_CHECK(max30101_write(handle, MAX30101_REG_FIFO_WR_PTR, 0x00)); // Set write pointer to 0 (optional)
    ESP_ERROR_CHECK(max30101_write(handle, MAX30101_REG_OVF_COUNTER, 0x00)); // Reset overflow counter
    ESP_ERROR_CHECK(max30101_write(handle, MAX30101_REG_FIFO_RD_PTR, 0x00)); // Reset read pointer
    ESP_ERROR_CHECK(max30101_write(handle, MAX30101_REG_SPO2_CONFIG, 0x27)); // Set SPO2 Configuration
    ESP_ERROR_CHECK(max30101_write(handle, MAX30101_REG_LED1_PA, 0x24)); // Pulse 7.2mA - adjustable LED
    ESP_ERROR_CHECK(max30101_write(handle, MAX30101_REG_LED2_PA, 0x24)); // Pulse 7.2mA - adjustable LED
    ESP_ERROR_CHECK(max30101_write(handle, MAX30101_REG_MODE_CONFIG, 0x02)); // Set to HR only
    return ESP_OK;
}

bool max30101_read_ir(i2c_master_dev_handle_t handle, uint32_t *ir_value)
{
    uint8_t write_pointer;
    uint8_t read_pointer;
    uint8_t fifo_data[3];

    if (ir_value == NULL) {
        return false;
    }

    if (max30101_read(handle, MAX30101_REG_FIFO_WR_PTR, &write_pointer, 1) != ESP_OK ||
        max30101_read(handle, MAX30101_REG_FIFO_RD_PTR, &read_pointer, 1) != ESP_OK ||
        write_pointer == read_pointer) {
        return false;
    }

    if (max30101_read(handle, MAX30101_REG_FIFO_DATA, fifo_data, sizeof(fifo_data)) != ESP_OK) {
        return false;
    }

    *ir_value = ((uint32_t)(fifo_data[0] & 0x03) << 16) |
                ((uint32_t)fifo_data[1] << 8) | fifo_data[2];
    return true;
}