idf_component_register(
	SRCS "main.c" "max30102.c" "sh1107.c"
	INCLUDE_DIRS "../include"
	REQUIRES esp_driver_i2c esp_driver_spi log esp_timer
)
