#ifndef BPNN_H
#define BPNN_H

// Define structures to represent the neural network
typedef struct {
    double *values;
    int size;
} Vector;
typedef struct{
    float pos;
    float speed;
    float theta;
    float omega;
} state;
typedef struct {
    Vector input;
    Vector hidden;
    Vector hidden_2;
    Vector output;
    Vector delta_hidden;
    Vector delta_hidden_2;
    Vector delta_output;
    Vector constant_input;
    Vector constant_hidden;
    Vector constant_hidden_2;
    double **input_hidden_weights;
    double **previous_input_hidden_weights;
    double **hidden_output_weights;
    double **hidden_hidden_weights;
    double **previous_hidden_hidden_weights;
    double **previous_hidden_output_weights;
    double learning_rate;
    double momentum;
    int initialized;
} BPNN;


// Function prototypes
void init_bpnn(BPNN *bpnn);
void cleanup_bpnn(BPNN *bpnn);
void initialize_weights(BPNN *bpnn);
double sigmoid(double x);
double sigmoid_derivative(double x);
double ReLU(double x);
double ReLU_derivative(double x);
double tanh_activation(double x);
double tanh_derivative(double x);
void compute(BPNN *bpnn, const double *input);
double learn(BPNN *bpnn, const double *input, const double *target);
double learn_all(BPNN *bpnn, const double **input, const double **target, int len, int times);
// double maxQValue(BPNN *bpnn, state s);
int greedypolicy(BPNN *bpnn,state s);

#endif