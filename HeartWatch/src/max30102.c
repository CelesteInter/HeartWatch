#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "max30102.h"

// Tag for bus
static const char *TAG = "max30102";

// Writes value to the specified max30102 register 
static esp_err_t max30102_write(i2c_master_dev_handle_t handle,
                                uint8_t reg, uint8_t value)
{
    uint8_t data[] = {reg, value};
    return i2c_master_transmit(handle, data, sizeof(data), -1);
}

// read length data into data pointer from register
static esp_err_t max30102_read(i2c_master_dev_handle_t handle,
                               uint8_t reg, uint8_t *data, size_t length)
{
    return i2c_master_transmit_receive(handle, &reg, 1, data, length, -1);
}

// initialize function for the max30102 using handle
esp_err_t max30102_init(i2c_master_dev_handle_t handle)
{
    uint8_t part_id;
    esp_err_t result = max30102_read(handle, MAX30102_REG_PART_ID, &part_id, 1);
    if (result != ESP_OK) {
        return result;
    }
    if (part_id != MAX30102_PART_ID) {
        ESP_LOGE(TAG, "Unexpected MAX30102 part ID: 0x%02X", part_id);
        return ESP_ERR_NOT_FOUND;
    }

    ESP_ERROR_CHECK(max30102_write(handle, MAX30102_REG_MODE_CONFIG, 0x40));
    vTaskDelay(pdMS_TO_TICKS(10));
    ESP_ERROR_CHECK(max30102_write(handle, MAX30102_REG_FIFO_WR_PTR, 0x00));
    ESP_ERROR_CHECK(max30102_write(handle, MAX30102_REG_OVF_COUNTER, 0x00));
    ESP_ERROR_CHECK(max30102_write(handle, MAX30102_REG_FIFO_RD_PTR, 0x00));
    ESP_ERROR_CHECK(max30102_write(handle, MAX30102_REG_SPO2_CONFIG, 0x27));
    ESP_ERROR_CHECK(max30102_write(handle, MAX30102_REG_LED1_PA, 0x24));
    ESP_ERROR_CHECK(max30102_write(handle, MAX30102_REG_LED2_PA, 0x24));
    ESP_ERROR_CHECK(max30102_write(handle, MAX30102_REG_MODE_CONFIG, 0x03));
    return ESP_OK;
}

// filter function, determines if IR value is valid (true) or default/empty (false)
// Reassigns ir_value if the read is valid
bool max30102_read_ir(i2c_master_dev_handle_t handle, uint32_t *ir_value)
{
    uint8_t write_pointer;
    uint8_t read_pointer;
    uint8_t fifo_data[6];

    if (max30102_read(handle, MAX30102_REG_FIFO_WR_PTR, &write_pointer, 1) != ESP_OK ||
        max30102_read(handle, MAX30102_REG_FIFO_RD_PTR, &read_pointer, 1) != ESP_OK ||
        write_pointer == read_pointer) {
        return false;
    }

    if (max30102_read(handle, MAX30102_REG_FIFO_DATA, fifo_data, sizeof(fifo_data)) != ESP_OK) {
        return false;
    }

    *ir_value = ((uint32_t)(fifo_data[3] & 0x03) << 16) |
                ((uint32_t)fifo_data[4] << 8) | fifo_data[5];
    return true;
}