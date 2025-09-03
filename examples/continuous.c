#include "CNN.h"
#include "deeplearning.h"
double avgRewards[] = {-1.7535404105646706, -0.3792807464971357, -0.022673601682935196, 0.2877342524833373, -2.219959456072523, 2.6369774068070644, 1.9621168025564124, 0.8283681617215288, -1.6972378121433693, 0.5236122759210953};
uint8_t k = 10;

double calculateReward(uint16_t choice) { //modified from online
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) {return calculateReward(choice); }
    double c = sqrt(-2 * log(r) / r);
    return u * c + avgRewards[choice];
}

int main() {
    srand(time(NULL));
    double* expectedRewards = DEEP_InitValues(10);
    uint32_t countPrevSelected[10] = {0,0,0,0,0,0,0,0,0,0};
    
    for (uint32_t t = 0; t < INT32_MAX; t++) {
        uint16_t choice = DEEP_UpperConfidenceBound(expectedRewards, countPrevSelected, 10, t+1, 2);
        double reward = calculateReward(choice);
        DEEP_UpdateValues(expectedRewards, choice, reward, 0.1);
        if (t % 10000000 == 0) {
            printf("%f%% done\n", (double)t/(double)INT32_MAX*100);
        }
    }
    printArray(expectedRewards, 10);
    free(expectedRewards);
    return 0;
}