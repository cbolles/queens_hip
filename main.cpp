#include <cstdint>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <bits/stdc++.h>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 512
#define THREADS_PER_BLOCK 256
#define BOARD_SIZE 8

using namespace std;

typedef struct individual_s {
    uint16_t fitness;
    uint8_t queensPosition[BOARD_SIZE];
} individual;


// Copied from HIP bit_extrack sample
#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }


/**
 * Handles initializing the population with random locations for the queens. 
 *
 * @param population The representation of the solutions to populate with random values
 * @param populationSize The number of individuals in the population
 */
void initPopulation(individual *population, uint16_t populationSize) {
    for(uint16_t individualIndex = 0; individualIndex < populationSize; individualIndex++) {
        for(uint8_t rowIndex = 0; rowIndex < BOARD_SIZE; rowIndex++) {
            // A queen can show up anywhere on the row from index 0 up to the size of the board
            uint8_t randomQueenIndex = rand() % BOARD_SIZE;
            cout << randomQueenIndex << endl;
            population[individualIndex].queensPosition[rowIndex] = randomQueenIndex;
            population[individualIndex].fitness = 0;
        }
    }
}

/**
 * Calculate the fitness for an individual. The fitness is the number of pairs of queens that
 * do not collide with each other. The max fitness is therefore the sum of the sequence from
 * 1...(n-1) = (n - 1)(n - 2)/2
 *
 * @param population The individuals to calculate the fitness for
 * @param populationSize The number of individuals
 */
__global__
void calculateFitness(individual *population, uint16_t populationSize) {
    // Determine the individual to calculate the fitness for
    size_t individualIndex = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    

    // Make sure we are within the bounds of the population count
    if(individualIndex >= populationSize) { return; }

    individual ind = population[individualIndex];
    // Assume max fitness, then decrease fitness based on collisions
    uint16_t fitness = ((BOARD_SIZE - 1) * (BOARD_SIZE - 2)) / 2;
    #pragma unroll
    for(int rowIndex = 0; rowIndex < BOARD_SIZE; rowIndex++) {
        uint8_t queenPosition = population[individualIndex].queensPosition[rowIndex];

        // Check each other queen location and check for a collision
        #pragma unroll
        for(int positionCheck = rowIndex + 1; positionCheck < BOARD_SIZE - 1; positionCheck++) {
            uint8_t otherQueenPosition = population[individualIndex].queensPosition[positionCheck];

            // If the queens are in the same column, that is a collision
            if(queenPosition == otherQueenPosition) {
                fitness--;
            }

            // If the queens are diagonal from each other, that is also a problem
            if(abs(queenPosition - otherQueenPosition) == abs(rowIndex - positionCheck)) {
                fitness--;
            }
        }
    }
    // Update fitness
    population[individualIndex].fitness = fitness;
}

/**
 * Used to compare and sort individuals in decenting order based on fitness
 */
bool compareIndividuals(individual first, individual second) {
    return first.fitness > second.fitness;
}

int main() {
    cout << "Being N-Queens Solver" << endl;

    // Size of the population during each generation
    uint16_t populationSize = 100;
    // Maximum number of generations to go through before ending
    uint16_t maxGenerations = 200;

    const uint16_t targetFitness = ((BOARD_SIZE - 1) * (BOARD_SIZE - 2)) / 2;

    // Initialize randomization
    srand(time(NULL));

    // Initialize the popultation with initially random data
    individual *h_population = static_cast<individual*>(calloc(populationSize, sizeof(individual)));
    initPopulation(h_population, populationSize);

    // Initialize device variables
    individual *d_population;
    CHECK(hipMalloc(&d_population, populationSize * sizeof(individual)));

    for(uint16_t populationId = 0; populationId < maxGenerations; populationId++) {
        // Calculate the fitness of the population
        CHECK(hipMemcpy(d_population, h_population, populationSize * sizeof(individual), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(calculateFitness, dim3(BLOCK_SIZE), dim3(THREADS_PER_BLOCK), 0, 0, d_population, populationSize);
        CHECK(hipMemcpy(h_population, d_population, populationSize * sizeof(individual), hipMemcpyDeviceToHost));

        // Sort the population based on fitness
        sort(h_population, h_population + populationSize, compareIndividuals);

        // Display the current highest fitness
        cout << h_population[0].fitness << endl;
        //cout << h_population[0].queensPosition[0] << endl;

        // Check to see if the end condition has been reached

        // Run through reproduction
    }

    return 0;
}
