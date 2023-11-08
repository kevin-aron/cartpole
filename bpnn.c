#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"bpnn.h"

#define NUM_IN 4
#define NUM_HID 4
#define NUM_HID_2 4
#define NUM_OUT 2
//Q-learning
#define NUM_ACTIONS 2
#define GAMMA 0.99
#define EPSILON 0.1
#define LEARNING_RATE 0.001

void init_vector(Vector *vector, int size) {
    vector->values = (double *)malloc(size * sizeof(double));
    vector->size = size;
}

void cleanup_vector(Vector *vector) {
    free(vector->values);
    vector->values = NULL;
    vector->size = 0;
}

void init_bpnn(BPNN *bpnn) {
    bpnn->input.values = NULL;
    bpnn->hidden.values = NULL;
    bpnn->hidden_2.values = NULL;
    bpnn->output.values = NULL;
    bpnn->delta_hidden.values = NULL;
    bpnn->delta_hidden_2.values = NULL;
    bpnn->delta_output.values = NULL;
    bpnn->constant_input.values = NULL;
    bpnn->constant_hidden.values = NULL;
    bpnn->constant_hidden_2.values = NULL;
    bpnn->input_hidden_weights = NULL;
    bpnn->previous_input_hidden_weights = NULL;
    bpnn->hidden_hidden_weights = NULL;
    bpnn->previous_hidden_hidden_weights = NULL;
    bpnn->hidden_output_weights = NULL;
    bpnn->previous_hidden_output_weights = NULL;
    bpnn->learning_rate = 0.001;
    bpnn->momentum = 0.3;
    bpnn->initialized = 0;

    // Initialize vectors
    init_vector(&(bpnn->input), NUM_IN);
    init_vector(&(bpnn->hidden), NUM_HID);
    init_vector(&(bpnn->hidden_2), NUM_HID_2);
    init_vector(&(bpnn->output), NUM_OUT);
    init_vector(&(bpnn->delta_hidden), NUM_HID);
    init_vector(&(bpnn->delta_hidden_2),NUM_HID_2);
    init_vector(&(bpnn->delta_output), NUM_OUT);
    init_vector(&(bpnn->constant_input), NUM_HID);
    init_vector(&(bpnn->constant_hidden), NUM_OUT);
    init_vector(&(bpnn->constant_hidden_2), NUM_HID_2);

    // Initialize weight matrices
    bpnn->input_hidden_weights = (double **)malloc(NUM_HID * sizeof(double *));
    bpnn->previous_input_hidden_weights = (double **)malloc(NUM_HID * sizeof(double *));
    bpnn->hidden_hidden_weights = (double **)malloc(NUM_HID_2*sizeof(double *));
    bpnn->previous_hidden_hidden_weights = (double **)malloc(NUM_HID_2 * sizeof(double *));
    bpnn->hidden_output_weights = (double **)malloc(NUM_OUT * sizeof(double *));
    bpnn->previous_hidden_output_weights = (double **)malloc(NUM_OUT * sizeof(double *));
    
    for (int i = 0; i < NUM_HID; i++) {
        bpnn->input_hidden_weights[i] = (double *)malloc(NUM_IN * sizeof(double));
        bpnn->previous_input_hidden_weights[i] = (double *)malloc(NUM_IN * sizeof(double));
    }
    for (int i = 0; i < NUM_HID_2; i++) {
        bpnn->hidden_hidden_weights[i] = (double *)malloc(NUM_HID * sizeof(double));
        bpnn->previous_hidden_hidden_weights[i] = (double *)malloc(NUM_HID * sizeof(double));
    }
    for (int i = 0; i < NUM_OUT; i++) {
        bpnn->hidden_output_weights[i] = (double *)malloc(NUM_HID * sizeof(double));
        bpnn->previous_hidden_output_weights[i] = (double *)malloc(NUM_HID * sizeof(double));
    }

    // Initialize weights
    initialize_weights(bpnn);
    bpnn->initialized = 1;
}

void cleanup_bpnn(BPNN *bpnn) {
    if (!bpnn->initialized)
        return;

    // Cleanup vectors
    cleanup_vector(&(bpnn->input));
    cleanup_vector(&(bpnn->hidden));
    cleanup_vector(&(bpnn->hidden_2));
    cleanup_vector(&(bpnn->output));
    cleanup_vector(&(bpnn->delta_hidden));
    cleanup_vector(&(bpnn->delta_hidden_2));
    cleanup_vector(&(bpnn->delta_output));
    cleanup_vector(&(bpnn->constant_input));
    cleanup_vector(&(bpnn->constant_hidden));
    cleanup_vector(&(bpnn->constant_hidden_2));

    // Cleanup weight matrices
    for (int i = 0; i < NUM_HID; i++) {
        free(bpnn->input_hidden_weights[i]);
        free(bpnn->previous_input_hidden_weights[i]);
    }
    free(bpnn->input_hidden_weights);
    free(bpnn->previous_input_hidden_weights);

    for (int i = 0; i < NUM_HID_2; i++) {
        free(bpnn->hidden_hidden_weights[i]);
        free(bpnn->previous_hidden_hidden_weights[i]);
    }
    free(bpnn->hidden_hidden_weights);
    free(bpnn->previous_hidden_hidden_weights);

    for (int i = 0; i < NUM_OUT; i++) {
        free(bpnn->hidden_output_weights[i]);
        free(bpnn->previous_hidden_output_weights[i]);
    }
    free(bpnn->hidden_output_weights);
    free(bpnn->previous_hidden_output_weights);

    bpnn->initialized = 0;
}

void initialize_weights(BPNN *bpnn) {
    for (int i = 0; i < NUM_HID; i++) {
        bpnn->constant_input.values[i] = (rand() / (double)RAND_MAX) / 2.0 - 0.25;
        for (int j = 0; j < NUM_IN; j++) {
            bpnn->input_hidden_weights[i][j] = (rand() / (double)RAND_MAX) / 2.0 - 0.25;
        }
    }
    for (int i = 0; i < NUM_HID_2; i++) {
        bpnn->constant_hidden_2.values[i] = (rand() / (double)RAND_MAX) / 2.0 - 0.25;
        for (int j = 0; j < NUM_HID; j++) {
            bpnn->hidden_hidden_weights[i][j] = (rand() / (double)RAND_MAX) / 2.0 - 0.25;
        }
    }
    for (int i = 0; i < NUM_OUT; i++) {
        bpnn->constant_hidden.values[i] = (rand() / (double)RAND_MAX) / 2.0 - 0.25;
        for (int j = 0; j < NUM_HID_2; j++) {
            bpnn->hidden_output_weights[i][j] = (rand() / (double)RAND_MAX) / 2.0 - 0.25;
        }
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}
double ReLU(double x){
    return (x>=0?x:(0.001*x));
}
double ReLU_derivative(double x){
    if(x<0) return 0.001;
    return 1.0;
}
double tanh_activation(double x){
    return tanh(x);
}
double tanh_derivative(double x){
    double tanh_x=tanh(x);
    return 1.0-tanh_x*tanh_x;
}

void compute(BPNN *bpnn, const double *input) {
    if (!bpnn->initialized) {
        exit(1); // Not initialized
    }
    for (int i = 0; i < NUM_IN; i++) {
        bpnn->input.values[i] = input[i];
    }
    // Initialize hidden and output vectors to zeros
    for (int i = 0; i < NUM_HID; i++) {
        bpnn->hidden.values[i] = 0.0;
    }
    for (int i = 0; i < NUM_HID_2; i++) {
        bpnn->hidden_2.values[i] = 0.0;
    }
    for (int i = 0; i < NUM_OUT; i++) {
        bpnn->output.values[i] = 0.0;
    }
    // Compute the hidden layer values
    for (int i = 0; i < NUM_HID; i++) {
        for (int j = 0; j < NUM_IN; j++) {
            bpnn->hidden.values[i] += bpnn->input_hidden_weights[i][j] * bpnn->input.values[j];
        }
        bpnn->hidden.values[i] = ReLU(bpnn->hidden.values[i] + bpnn->constant_input.values[i]);
    }
    for (int i = 0; i < NUM_HID_2; i++) {
        for (int j = 0; j < NUM_HID; j++) {
            bpnn->hidden_2.values[i] += bpnn->hidden_hidden_weights[i][j] * bpnn->hidden.values[j];
        }
        bpnn->hidden_2.values[i] = ReLU(bpnn->hidden_2.values[i] + bpnn->constant_hidden.values[i]);
    }
    // Compute the output layer values
    for (int i = 0; i < NUM_OUT; i++) {
        for (int j = 0; j < NUM_HID_2; j++) {
            bpnn->output.values[i] += bpnn->hidden_output_weights[i][j] * bpnn->hidden_2.values[j];
        }
        bpnn->output.values[i] = tanh_activation(bpnn->output.values[i] + bpnn->constant_hidden_2.values[i]);
    }
}

double learn(BPNN *bpnn, const double *input, const double *target) {
    // Compute the network's output for the given input
    compute(bpnn, input);

    // Compute the errors and deltas
    //gai target jisuan Qvalue 
    double error = 0.0;
    for (int i = 0; i < NUM_OUT; i++) {
        bpnn->delta_output.values[i]=tanh_derivative(bpnn->output.values[i])*(target[i]-bpnn->output.values[i]);
        error += fabs(bpnn->delta_output.values[i]);
    }
    for (int i = 0; i < NUM_HID_2; i++) {
        bpnn->delta_hidden_2.values[i] = 0.0;
        for (int j = 0; j < NUM_OUT; j++) {
            bpnn->delta_hidden_2.values[i] += bpnn->hidden_output_weights[j][i] * bpnn->delta_output.values[j];
        }
        bpnn->delta_hidden_2.values[i] *= ReLU_derivative(bpnn->hidden_2.values[i]);
        error += fabs(bpnn->delta_hidden_2.values[i]);
    }
    for (int i = 0; i < NUM_HID; i++) {
        bpnn->delta_hidden.values[i] = 0.0;
        for (int j = 0; j < NUM_HID_2; j++) {
            bpnn->delta_hidden.values[i] += bpnn->hidden_hidden_weights[j][i] * bpnn->delta_hidden_2.values[j];
        }
        bpnn->delta_hidden.values[i] *= ReLU_derivative(bpnn->hidden.values[i]);
        error += fabs(bpnn->delta_hidden.values[i]);
    }
    // Update the network's weights
    double d_ij;
    for (int i = 0; i < NUM_OUT; i++) {
        for (int j = 0; j < NUM_HID_2; j++) {
            d_ij = bpnn->learning_rate * bpnn->delta_output.values[i] * bpnn->hidden_2.values[j] +
                   bpnn->momentum * bpnn->previous_hidden_output_weights[i][j];
            bpnn->hidden_output_weights[i][j] += d_ij;
            bpnn->previous_hidden_output_weights[i][j] = d_ij;
        }
        bpnn->constant_hidden_2.values[i] += bpnn->learning_rate * bpnn->delta_output.values[i];
    }
    for (int i = 0; i < NUM_HID_2; i++) {
        for (int j = 0; j < NUM_HID; j++) {
            d_ij = bpnn->learning_rate * bpnn->delta_hidden_2.values[i] * bpnn->hidden.values[j] +
                   bpnn->momentum * bpnn->previous_hidden_hidden_weights[i][j];
            bpnn->hidden_hidden_weights[i][j] += d_ij;
            bpnn->previous_hidden_hidden_weights[i][j] = d_ij;
        }
        bpnn->constant_hidden.values[i] += bpnn->learning_rate * bpnn->delta_hidden_2.values[i];
    }
    for (int i = 0; i < NUM_HID; i++) {
        for (int j = 0; j < NUM_IN; j++) {
            d_ij = bpnn->learning_rate * bpnn->delta_hidden.values[i] * bpnn->input.values[j] +
                   bpnn->momentum * bpnn->previous_input_hidden_weights[i][j];
            bpnn->input_hidden_weights[i][j] += d_ij;
            bpnn->previous_input_hidden_weights[i][j] = d_ij;
        }
        bpnn->constant_input.values[i] += bpnn->learning_rate * bpnn->delta_hidden.values[i];
    }
    return error;
}

double learn_all(BPNN *bpnn, const double **input, const double **target, int len,int times) {
    double sumerr = 0.0;
    for (int i = 0; i < times; i++) {
        sumerr = 0.0;
        for (int j = 0; j < len; j++) {
            sumerr += learn(bpnn, input[j], target[j]);
        }
        printf("%d:\t%f\n", i, sumerr);
    }
    return sumerr;
}

// double maxQValue(BPNN *bpnn, state s){
//     double qvalue[NUM_ACTIONS];
//     double input_data[4] = {t_weight[0]*s.pos,t_weight[1]*s.speed,t_weight[2]*s.theta,t_weight[3]*s.omega};
//     compute(bpnn,input_data);
//     qvalue[0]=bpnn->output.values[0];
//     qvalue[1]=bpnn->output.values[1];
//     printf("left:%lf\tright:%lf\n",qvalue[0],qvalue[1]);
//     return (qvalue[0]>qvalue[1]?qvalue[0]:qvalue[1]);
// }
int greedypolicy(BPNN *bpnn,state s){
    srand(time(NULL));
    if((double)rand()/RAND_MAX < EPSILON){
        return rand()%NUM_ACTIONS;
    }
    double input_data[4]={s.pos,s.speed,s.theta,s.omega};
    compute(bpnn,input_data);
    return (bpnn->output.values[0] > bpnn->output.values[1])?0:1;
}