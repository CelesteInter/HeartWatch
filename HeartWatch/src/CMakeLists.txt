idf_component_register(
	SRCS "main.c" "max30101.c" "hud.c" "mpu.c"
	INCLUDE_DIRS "../include"
	REQUIRES esp_driver_i2c esp_driver_spi esp_lcd log esp_timer
)
