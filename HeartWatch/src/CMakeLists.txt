idf_component_register(
	SRCS "main.c" "max30102.c"
	INCLUDE_DIRS "../include"
	REQUIRES esp_driver_i2c log esp_timer
)
