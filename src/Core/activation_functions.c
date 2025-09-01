#include "headers.h"

void CNN_ApplyAF(double* input, u_int16_t size, enum CNN_ActivationFunctions function)
{
    switch (function) 
    {
        case LeakyReLU:
            CNN_LeakyReLU(input, size);
            break;
        case ReLU:
            CNN_ReLU(input, size);
            break;
        case Sigmoid:
            CNN_Sigmoid(input, size);
            break;
        case Softmax:
            CNN_Softmax(input, size);
            break;
        case Linear:
            CNN_Linear(input, size);
            break;
    }
}

void CNN_ApplyAFDerivative(double* input, u_int16_t size, enum CNN_ActivationFunctions function) 
{
    switch (function) 
    {
        case LeakyReLU:
            CNN_dLeakyReLU(input, size);
            break;
        case ReLU:
            CNN_dReLU(input, size);
            break;
        case Sigmoid:
            CNN_dSigmoid(input, size);
            break;
        case Softmax:
            CNN_dSoftmax(input, size);
            break;
        case Linear:
            CNN_dLinear(input, size);
            break;
    }
}

void CNN_Linear(double* inp, uint16_t size) { }

void CNN_dLinear(double* inp, uint16_t size) 
{
    for (uint16_t i = 0; i < size; i++) 
    {
        inp[i] = 1;
    }
}

void CNN_LeakyReLU(double* inp, uint16_t size)
{
    for (uint16_t i = 0; i < size; i++)
    {
        inp[i] = inp[i]>0?inp[i]:inp[i]*0.1;
    }
}

void CNN_dLeakyReLU(double* inp, uint16_t size)
{
    for (uint16_t i = 0; i < size; i++)
    {
        inp[i] = inp[i]>0?1:.1;
    }
}

void CNN_ReLU(double* inp, uint16_t size) 
{
    for (uint16_t i = 0; i < size; i++) 
    {
        inp[i] = inp[i]>0?inp[i]:0;
    }
}

void CNN_dReLU(double* inp, uint16_t size)
{
    for (uint16_t i = 0; i < size; i++) 
    {
        inp[i] = inp[i]>0?1:0;
    }
}

void CNN_Sigmoid(double* inp, uint16_t size) 
{
    for (uint16_t i = 0; i < size; i++) 
    {
        inp[i] = 1/(1+exp(-inp[i]));
    }
}

void CNN_dSigmoid(double* inp, uint16_t size) 
{
    CNN_Sigmoid(inp, size);
    for (uint16_t i = 0; i < size; i++) 
    {
        inp[i] = inp[i] * (1 - inp[i]);
    }
}

void CNN_Softmax(double *inp, uint16_t size) 
{
    double m = -INFINITY;

    for (size_t i = 0; i < size; i++) 
    {
        if (inp[i] > m)
            m = inp[i];
    }

    double sum = 0;
    for (size_t i = 0; i < size; i++) 
        sum += expf(inp[i] - m);

    double offset = m + logf(sum);
    for (size_t i = 0; i < size; i++) 
        inp[i] = expf(inp[i] - offset);
}

void CNN_dSoftmax(double* inp, uint16_t size) 
{
    double sum = 0;

    double d = 0.001;
    double* adjusted = (double*)malloc(sizeof(double) * size);
    double* original = (double*)malloc(sizeof(double) * size);
    double* output = (double*)malloc(sizeof(double) * size);
    memcpy(original, inp, sizeof(double) * size);
    CNN_Softmax(original, size);

    for (uint16_t i = 0; i < size; i++) 
    {
        memcpy(adjusted, inp, sizeof(double) * size);

        adjusted[i] += d;

        CNN_Softmax(adjusted, size);
        output[i] = (adjusted[i]-original[i])/d;
    }
    memcpy(inp, output, sizeof(double)*size);
}