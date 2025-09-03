#include "CNN.h"

void randomArray(double* arr, uint16_t n);
void sumArrays(double* a1, double* a2, uint16_t c);
void dotProduct(double* input, double** weights, uint16_t nCount, uint16_t wPerNeuron);
double* mutateArray(uint16_t n, double* arr, double mutationRate);
double max(double* arr, double c);

void printArray(double* arr, uint16_t n);
void printByteArray(uint16_t* arr, uint16_t n);
void normalize(double* arr, uint16_t n);
void scale(double* arr, uint16_t n, double scaler);