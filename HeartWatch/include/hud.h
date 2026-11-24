/**
 * Header code for the HUD of the Display
 */

#include <time.h>
#include <stdbool.h>
#include <stdint.h>

typedef enum {
    SCREEN_BOOTUP = 0,
    SCREEN_HOME = 1,
    SCREEN_ACTIVE = 2,
    SCREEN_TRANSMISSION = 3,
    SCREEN_COUNT
} watch_screen_t;

/**
 * Initializes LVGL Screens
 */
void setupScreens();

void displayBootup();

void displayHome();

void displayStopwatch();

void displayActive();

void displayTransmission();

void updateHomeScreen(const struct tm *timeinfo, uint8_t battery_pct);

void updateActiveScreen(uint16_t bpm, bool is_moving);

/**
 * Displays the selected screen
 * 0 = Bootup Screen
 * 1 = Time-telling display
 * 2 = Heartrate Sensing Display
 * 3 = Motion Sensing Screen (IMU)
 */
 //void displayHUD(int selection);


