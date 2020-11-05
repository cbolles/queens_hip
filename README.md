# Queens with HIP

This is a side project to experiment with using HIP on AMD devices. This is a project working with
the N-queens problem handled using a genetic algorithm. The genetic algorithm lends itself well
to acceleration benefits at several steps.

Currently only the fitness calculations make use of the GPU, however the goal is to move more of
the computation over to the GPU.

Testing is incomplete and the code is not yet finished.