// Include libraries here
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Macros for GPIO
#define USER_LED GPIO_NUM_21

// Main application loop
void app_main(void)
{
    // Config
	gpio_config_t led_config = {
		.pin_bit_mask = 1ULL << USER_LED,
		.mode = GPIO_MODE_OUTPUT,
		.pull_up_en = GPIO_PULLUP_DISABLE,
		.pull_down_en = GPIO_PULLDOWN_DISABLE,
		.intr_type = GPIO_INTR_DISABLE,
	};
	gpio_config(&led_config);

    // Main polling loop
	while (1) {
		gpio_set_level(USER_LED, 0);
		vTaskDelay(pdMS_TO_TICKS(500));
		gpio_set_level(USER_LED, 1);
		vTaskDelay(pdMS_TO_TICKS(500));
	}
}

