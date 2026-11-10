#include "lvgl.h"

// Flush callback: LVGL passes rendered pixel coordinates and a pixel buffer
void my_disp_flush(lv_disp_drv_t *disp_drv, const lv_area_t *area, lv_color_t *color_p) {
    uint32_t w = (area->x2 - area->x1 + 1);
    uint32_t h = (area->y2 - area->y1 + 1);

    // Send pixel buffer over SPI to display hardware using your screen's driver
    lcd_draw_bitmap(area->x1, area->y1, w, h, (uint16_t *)color_p);

    // Tell LVGL that the transfer is complete so it can unlock the buffer
    lv_disp_flush_ready(disp_drv);
}


