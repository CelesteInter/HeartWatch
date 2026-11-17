#include "esp_log.h"
#include "mpu.h"

static const char *TAG = "mpu";
static i2c_master_dev_handle_t device_handle;
static bool initialized;

static esp_err_t mpu_read_register(uint8_t reg, uint8_t *data, size_t length)
{
    return i2c_master_transmit_receive(
        device_handle, &reg, 1, data, length, -1);
}

bool mpu_init(i2c_master_bus_handle_t bus_handle)
{
    if (initialized) {
        return true;
    }
    if (bus_handle == NULL) {
        ESP_LOGE(TAG, "Cannot initialize MPU without an I2C bus");
        return false;
    }

    i2c_device_config_t device_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MPU_I2C_ADDRESS,
        .scl_speed_hz = 100000,
    };

    esp_err_t result = i2c_master_bus_add_device(
        bus_handle, &device_config, &device_handle);
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add MPU device");
        return false;
    }

    uint8_t who_am_i;
    result = mpu_read_register(MPU_REG_WHO_AM_I, &who_am_i, 1);
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "MPU did not respond");
        return false;
    }

    if (who_am_i != MPU_WHO_AM_I_VALUE) {
        ESP_LOGE(TAG, "Unexpected WHO_AM_I value: 0x%02X", who_am_i);
        return false;
    }

    initialized = true;
    ESP_LOGI(TAG, "MPU initialized: WHO_AM_I=0x%02X", who_am_i);
    return true;
}

bool mpu_read(int16_t *accel_x, int16_t *accel_y, int16_t *accel_z,
              int16_t *gyro_x, int16_t *gyro_y, int16_t *gyro_z)
{
    if (!initialized) {
        return false;
    }

    uint8_t data[MPU_DATA_LENGTH];
    if (mpu_read_register(MPU_REG_ACCEL_XOUT_H, data, sizeof(data)) != ESP_OK) {
        return false;
    }

    int16_t raw_accel_x = (int16_t)(((uint16_t)data[0] << 8) | data[1]);
    int16_t raw_accel_y = (int16_t)(((uint16_t)data[2] << 8) | data[3]);
    int16_t raw_accel_z = (int16_t)(((uint16_t)data[4] << 8) | data[5]);
    int16_t raw_gyro_x = (int16_t)(((uint16_t)data[8] << 8) | data[9]);
    int16_t raw_gyro_y = (int16_t)(((uint16_t)data[10] << 8) | data[11]);
    int16_t raw_gyro_z = (int16_t)(((uint16_t)data[12] << 8) | data[13]);

    if (accel_x != NULL) {
        *accel_x = raw_accel_x;
    }
    if (accel_y != NULL) {
        *accel_y = raw_accel_y;
    }
    if (accel_z != NULL) {
        *accel_z = raw_accel_z;
    }
    if (gyro_x != NULL) {
        *gyro_x = raw_gyro_x;
    }
    if (gyro_y != NULL) {
        *gyro_y = raw_gyro_y;
    }
    if (gyro_z != NULL) {
        *gyro_z = raw_gyro_z;
    }

    return true;
}

bool mpu_is_ready(void)
{
    uint8_t who_am_i;

    return initialized &&
           mpu_read_register(MPU_REG_WHO_AM_I, &who_am_i, 1) == ESP_OK &&
           who_am_i == MPU_WHO_AM_I_VALUE;
}
