#include <cstdint>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <bits/stdc++.h>
#include <hip/hip_runtime.h>
#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>

#define BLOCK_SIZE 512
#define THREADS_PER_BLOCK 256

using namespace std;

/**
 * Represents a single potential solution to the queens problem, a solution stores which column
 * a queen would be placed for each row. So for a board of size 10, the solution would have 10
 * values, each value would be the column to place the queen.
 */
struct individual {
    uint16_t fitness;
    uint8_t *queensPositions;
};

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
void initPopulation(individual *population, uint16_t populationSize, uint8_t boardSize) {
    for(uint16_t individualIndex = 0; individualIndex < populationSize; individualIndex++) {
        // Allocate space for all of the queen positions
        CHECK(hipMallocManaged(&population[individualIndex].queensPositions, boardSize * sizeof(uint8_t)));
        
        // Randomly assign each location for the queens
        for(uint8_t rowIndex = 0; rowIndex < boardSize; rowIndex++) {
            // A queen can show up anywhere on the row from index 0 up to the size of the board
            uint8_t randomQueenIndex = rand() % boardSize;
            population[individualIndex].queensPositions[rowIndex] = randomQueenIndex;
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
void calculateFitness(individual *population, uint16_t populationSize, uint8_t boardSize, uint16_t targetFitness) {
    // Determine the individual to calculate the fitness for
    size_t individualIndex = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    

    // Make sure we are within the bounds of the population count
    if(individualIndex >= populationSize) { return; }

    // Assume max fitness, then decrease fitness based on collisions
    uint16_t fitness = targetFitness;
    #pragma unroll
    for(int rowIndex = 0; rowIndex < boardSize; rowIndex++) {
        uint8_t queenPosition = population[individualIndex].queensPositions[rowIndex];
        #pragma unroll
        // Check each other queen location and check for a collision
        for(int positionCheck = rowIndex + 1; positionCheck < boardSize - 1; positionCheck++) {
            uint8_t otherQueenPosition = population[individualIndex].queensPositions[positionCheck];

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
 * Handles running the reproduction on the population. Assumes that the more fit
 * individuals are towards the start of the array and the less fit towards the end.
 * Once reproduction takes place, the population is updated with the next
 * generation. Currently the top 50% get to reproduce, this can be improved.
 * 
 * @param population The individuals to run reproduction on. Will be replaced with the next generation.
 * @param populationSize The number of individuals in the population.
 */
void reproduction(individual *population, uint16_t populationSize, uint8_t boardSize) {
    individual nextGen[populationSize];

    for(int i = 0; i < populationSize; i++) {
        // Use elitist approach of simply having the top 50% reproduce
        individual firstParent = population[rand() % (populationSize / 2)];
        individual secondParent = population[rand() % (populationSize / 2)];

        individual child;
        // Top half from parent 1, bottom half from parent 2
        int midPoint = boardSize / 2;
        for(int j = 0; j < midPoint; j++) {
            child.queensPositions[j] = firstParent.queensPositions[j];
            child.queensPositions[j + midPoint] = secondParent.queensPositions[j + midPoint];
        }

        // Random mutation chance
        if(rand() % 100 <= 5) {
            child.queensPositions[rand() % boardSize] = rand() % boardSize;
        }
        nextGen[i] = child;
    }

    // Copy over the next generation into the current generation
    // CHECK(hipMemcpy(population, nextGen, sizeof(individual) * populationSize, hipMemcpyHostToDevice));
}

/**
 * Used to compare and sort individuals in decenting order based on fitness
 */
bool compareIndividuals(individual first, individual second) {
    return first.fitness > second.fitness;
}

int main(int argc, char **argv) {
    // Get command line arguments
    CLI::App app("N-Queens Problem Solver");

    uint16_t populationSize = 1000;
    app.add_option("-p, --population", populationSize, "Number of individuals in each generation, defaults to 1000");

    uint16_t maxGenerations = -1;
    app.add_option("-m, --max", maxGenerations, "Maximum generations to run for, defaults to infinite");

    uint16_t boardSize = 8;
    app.add_option("-s, --size", boardSize, "Size of the board, default to 8");

    // Print some starting information
    cout << "Welcome to the N-Queens Solver" << endl;
    cout << "Population Size: " << populationSize << " Max Generations: " << maxGenerations << endl;

    CLI11_PARSE(app, argc, argv);

    // Initialize randomization
    srand(time(NULL));

    // Initialize the popultation with initially random data
    individual *population;
    CHECK(hipMallocManaged(&population, populationSize * sizeof(individual)));
    initPopulation(population, populationSize, boardSize);

    // Calculate the target fitness
    const uint16_t targetFitness = ((boardSize) * (boardSize - 1)) / 2;

    for(uint16_t populationId = 0; 1; populationId++) {
        // Calculate the fitness of the population
        hipLaunchKernelGGL(calculateFitness, dim3(BLOCK_SIZE), dim3(THREADS_PER_BLOCK), 0, 0, population, populationSize, boardSize, targetFitness);

        hipDeviceSynchronize();

        // Sort the population based on fitness
        sort(population, population + populationSize, compareIndividuals);

        // Display the current highest fitness
        cout << "Generation: " << populationId << " best fitness: " << population[0].fitness << " target: " << targetFitness << endl;

        // Check to see if the end condition has been reached
        if(population[0].fitness >= targetFitness) {
            break;
        }

        // Run through reproduction
        reproduction(population, populationSize, boardSize);
    }

    // Print if the ideal combination was found or not
    if(population[0].fitness >= targetFitness) {
        cout << "Ideal combination found!" << endl;
    } else {
        cout << "Could not find ideal combination" << endl;
    }

    // Print out the location to place the queens regardless of if it is the ideal configuration
    for(int i = 0; i < boardSize; i++) {
        cout << +population[0].queensPositions[i] << " ";
    }
    cout << endl;

    // Free unused variables
    hipFree(population);

    return 0;
}
