#include <cstdint>
#include <stdlib.h>
#include <time.h>

#include "hip/hip_runtime.h"

/**
 * Handles initializing the population with random locations for the queens. 
 *
 * @param population The representation of the solutions to populate with random values
 * @param populationSize The number of individuals in the population
 * @param boardSize The square dimension of the board
 */
void initPopulation(uint8_t **population, uint16_t populationSize, uint8_t boardSize) {
    for(uint16_t individualIndex = 0; individualIndex < populationSize; individualIndex++) {
        for(uint8_t rowIndex = 0; rowIndex < boardSize; rowIndex++) {
            // A queen can show up anywhere on the row from index 0 up to the size of the board
            uint8_t randomQueenIndex = rand() % boardSize;
            population[individualIndex][rowIndex] = randomQueenIndex;
        }
    }
}

/**
 * Calculate the fitness for an individual. The fitness is the number of pairs of queens that
 * do not collide with each other. The max fitness is therefore the sum of the sequence from
 * 1...(n-1) = (n - 1)(n - 2)/2
 *
 * @param population The individuals to calculate the fitness for
 * @param fitness The array to store the fitness values into
 * @param populationSize The number of individuals
 * @param boardSize The size of the board
 */
__global__
void calculateFitness(uint8_t **population, uint8_t *fitness, uint16_t populationSize, uint8_t boardSize) {
    // Determine the individual to calculate the fitness for
    size_t individualIndex = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // Make sure we are within the bounds of the population count
    if(individualIndex >= populationSize) { return; }

    uint8_t *individual = population[individualIndex];
    // Assume max fitness, then decrease fitness based on collisions
    uint16_t fitness = ((boardSize - 1) * (boardSize - 2)) / 2;
    #pragma unroll
    for(int rowIndex = 0; rowIndex < boardSize; rowIndex++) {
        uint8_t queenPosition = individual[rowIndex];

        // Check each other queen location and check for a collision
        #pragma unroll
        for(int positionCheck = rowIndex + 1; positionCheck < boardSize - 1; positionCheck++) {
            uint8_t otherQueenPosition = individual[positionCheck];

            // If the queens are in the same column, that is a collision
            if(queenPosition == otherQueenPosition) {
                fitness--;
            }

            // If the queens are diagonal from each other, that is also a problem
            if(abs(queenPosition - otherQueenPosition) == abs(rowIndex - positionCheck)) {
                fitness--;
            }
        }
        // Update fitness
        fitness[individualIndex] = fitness;
    }
}

int main() {
    // Number of rows that the chess board will be
    uint8_t boardSize = 5;
    // Size of the population during each generation
    uint16_t h_populationSize = 100;
    // Maximum number of generations to go through before ending
    uint16_t maxGenerations = 200;

    // Initialize randomization
    srand(time(NULL));

    // Initialize the popultation with initially random data
    uint8_t **h_population = calloc(h_populationSize, sizeof(uint8_t) * boardSize);
    uint8_t *h_fitness = calloc(h_popultationSize, sizeof(uint16_t));

    // Initialize device variables
    uint8_t **d_population = hipMalloc(h_popultationSize * sizeof(uint8_t) * boardSize);
    uint8_t *h_fitness = hipMalloc(h_popultationSize * sizeof(uint16_t));

    for(uint16_t populationId = 0; populationId < maxGenerations; populationId++) {
        // Calculate the fitness of the population
         
        // Sort the population based on fitness
        
        // Display the current highest fitness

        // Check to see if the end condition has been reached

        // Run through reproduction
    }

    return 0;
}
