/**
 * Header code for the HUD of the Display
 */

#include <time.h>
#include <stdbool.h>
#include <stdint.h>

typedef enum {
    SCREEN_BOOTUP = 0,
    SCREEN_SELECTION = 1,
    SCREEN_HOME = 2,
    SCREEN_STOPWATCH = 3,
    SCREEN_ACTIVE = 4,
    SCREEN_TRANSMISSION = 5,
    SCREEN_COUNT = 6
} watch_screen_t;

/**
 * Initializes LVGL Screens
 */
void setupScreens();

void displayBootup();

void displaySelection();

void displayHome();

void displayStopwatch();

void displayActive();

void displayTransmission();

void updateHomeScreen(const struct tm *timeinfo, uint8_t battery_pct);

void updateSelectionScreen(watch_screen_t current_screen);

void updateStopwatchScreen(uint32_t elapsed_ms, bool is_running);

void updateActiveScreen(uint16_t bpm, bool is_logging);

void updateTransmissionScreen(bool is_transmitting);

/**
 * Displays the selected screen
 * 0 = Bootup Screen
 * 1 = Time-telling display
 * 2 = Heartrate Sensing Display
 * 3 = Motion Sensing Screen (IMU)
 */
 //void displayHUD(int selection);


