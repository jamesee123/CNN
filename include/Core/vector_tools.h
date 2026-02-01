#include "CNN.h"

/*
Initializes an array with random values between -1.5 and 1.5
*/
void randomArray(double* arr, uint16_t n);

/*
Adds the perameters of two arrays together and stores the result in the first array
*/
void sumArrays(double* a1, double* a2, uint16_t c);

/*
Runs the dot product on a 1D array and a 2D array and stores the results in the 1D array.
*/
void dotProduct(double* input, double** weights, uint16_t nCount, uint16_t wPerNeuron);

/*
Randomly adjusts an array's values with a magnitude based on the mutationRate perameter.
*/
double* mutateArray(uint16_t n, double* arr, double mutationRate);

/*
Returns the greatest value in an array
*/
double max(double* arr, double c);

/*
Displays an array
*/
void printArray(double* arr, uint16_t n);

/*
Displays the binary of an array
*/
void printByteArray(uint16_t* arr, uint16_t n);

/*
Sets the magnitude of an array to 1
*/
void normalize(double* arr, uint16_t n);

/*
Sets the magnitude of an array to the scaler value
*/
void scale(double* arr, uint16_t n, double scaler);