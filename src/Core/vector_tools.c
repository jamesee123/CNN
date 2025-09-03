#include "CNN.h"

void randomArray(double* output, uint16_t n) 
{
    for (uint16_t i = 0; i < n; i++) {
        output[i] = ((double)rand() / (double)RAND_MAX * 3)-(double)1.5;
    }
}

double max(double* arr, double c) 
{
    double biggest = 0;
    for (uint16_t i = 0; i < c; i++) {
        if (biggest < arr[i]) biggest = arr[i];
    }

    return biggest;
}

void sumArrays(double* a1, double* a2, uint16_t c) 
{
    for (uint16_t i = 0; i < c; i++) {
        a1[i] = a1[i] + a2[i];
    }
}

void dotProduct(double* input, double** weights, uint16_t nCount, uint16_t wPerNeuron) 
{
    double* output = malloc(sizeof(double) * nCount);
    for (uint16_t n = 0; n < nCount; n++) 
    {
        output[n] = 0;
        for (uint16_t i = 0; i < wPerNeuron; i++) 
        {
            output[n] += input[i] * weights[n][i];
        }
    }
    memcpy(input, output, nCount * sizeof(double));
}

double* mutateArray(uint16_t n, double* arr, double mutationRate) 
{
    double* toReturn = (double*)malloc(sizeof(double) * n);
    for (uint16_t i = 0; i < n; i++) 
    {
        toReturn[i] = (((double)rand() / (double)RAND_MAX)-.5) * mutationRate + arr[i];
    }

    return toReturn;
}

void printArray(double* arr, uint16_t n) 
{
    for (uint16_t i = 0; i < n; i++) printf("%f ", arr[i]);
    printf("\n");
    fflush(stdout);
}

void printByteArray(uint16_t* arr, uint16_t n) 
{
    for (uint16_t i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
    fflush(stdout);
}

double absVal(double x)
{
    return x>0?x:-x;
}

void scale(double* arr, uint16_t n, double scaler) 
{
    for (u_int16_t i = 0; i < n; i++) 
    {
        arr[i] *= scaler;
    }
}

void normalize(double* arr, uint16_t n) 
{
    double sum = 0;
    for (uint16_t i = 0; i < n; i++) {
        sum += absVal(arr[i]);
    }

    sum = sum==0?.0001:sum;
    for (uint16_t i = 0; i < n; i++) {
        arr[i] /= sum;
    }
}