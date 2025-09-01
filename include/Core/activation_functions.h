#include "headers.h"

enum CNN_ActivationFunctions 
{
    LeakyReLU,
    ReLU,
    Sigmoid,
    Softmax,
    Linear
};

void CNN_ApplyAFDerivative(double* input, u_int16_t size, enum CNN_ActivationFunctions function);
void CNN_ApplyAF(double* input, u_int16_t size, enum CNN_ActivationFunctions function);
void CNN_LeakyReLU(double* inp, uint16_t size);
void CNN_dLeakyReLU(double* inp, uint16_t size);
void CNN_ReLU(double* inp, uint16_t size);
void CNN_dReLU(double* inp, uint16_t size);
void CNN_Sigmoid(double* inp, uint16_t size);
void CNN_dSigmoid(double* inp, uint16_t size);
void CNN_Softmax(double* inp, uint16_t size);
void CNN_dSoftmax(double* inp, uint16_t size);
void CNN_Linear(double* inp, uint16_t size);
void CNN_dLinear(double* inp, uint16_t size);