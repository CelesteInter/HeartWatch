#include "lvgl.h"
#include "esp_lcd_panel_ops.h"

void my_disp_flush(lv_disp_drv_t *disp_drv, const lv_area_t *area, lv_color_t *color_p) {
    esp_lcd_panel_handle_t panel = (esp_lcd_panel_handle_t)disp_drv->user_data;

    // Push the rendered pixel buffer over SPI via esp_lcd DMA
    esp_lcd_panel_draw_bitmap(panel, area->x1, area->y1, area->x2 + 1, area->y2 + 1, color_p);
}