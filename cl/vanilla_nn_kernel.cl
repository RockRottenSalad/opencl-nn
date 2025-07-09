
float relu(const float x) {
    return (x > 0 ? x : 0);
}

float relu_prime(const float x) {
    return (x > 0 ? 1 : 0);
}

float sigmoid(const float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Note that the expected argument is sigmoid(x) and not x
float sigmoid_lazy_prime(const float sigmoid_x) {
    return sigmoid_x * (1.0 - sigmoid_x);
}

void kernel backprop_delta_init(
    global float* A,
    global float* out,
    const uint n)
{
    int id = get_global_id(0);

    if(id >= n) return;

    out[id] = A[id] - out[id];
}


// Update previous biases, weights and gA values which act as (aL - y) for the hidden layers
void kernel backprop_step(
    global float* W,
    global float* gW,
    global float* gB,
    global float* A,
    global float* prevA,
    global float* gA,
    global float* prevgA,
    const uint cols,
    const uint rows)
{
    int id = get_global_id(0);

    // Update gW and gB
    if(id < cols) {
        float aL = A[id];

        // (aL - y)
        float diff_ay = gA[id];

        // aL * (1 - aL)
        float activation_prime = sigmoid_lazy_prime(aL);

        float delta = 2.0 * diff_ay * activation_prime;

        gB[id] += delta;

        // rows = rows in W
        // cols = columns in W
        for(int k = 0; k < rows; k++) {
            float aLprev = prevA[k];
            gW[k*cols + id] += aLprev * delta;
        }
    }

    if(id >= rows) return;

    // Update gA
    for(int j = 0; j < cols; j++) {
        float activation_prime = sigmoid_lazy_prime(A[j]);
        float diff_ay = gA[j];
        float delta = 2.0 * diff_ay * activation_prime;

        prevgA[id] += W[id*cols + j] * delta;
    }
}

void kernel apply_gradient(
    global float* W,
    global float* gW,
    global float* B,
    global float* gB,
    const uint cols,
    const uint rows,
    const uint n
) {
    const int id = get_global_id(0);
    if(id >= cols) return;

    const float nf = (float)n;

    B[id] -= gB[id] / nf;
    for(int i = 0; i < rows; i++) W[i*cols + id] -= gW[i*cols + id] / nf;

}

void kernel cost(global float* A, global float* expected, const int n, global float* out) {
    // This doesn't run in parallel, not really needed
    if(get_global_id(0) != 0) return;

    for(int i = 0; i < n; i++) {
        float diff = A[0] - expected[0];
        out[0] += diff*diff;
    }

    out[0] /= n;
}

void kernel forward(
    global float* W,
    global float* B,
    global float* A,
    const uint rows,
    const uint cols,
    global float* out)
{ 
    int id = get_global_id(0);

    if(id >= cols) return;

    float value = 0;
    for(int i = 0; i < rows; i++) {
        value += A[i] * W[i*cols + id];
    }
    value += B[id];

    out[id] = sigmoid(value);
} 


// Equivalent to backprop_step, but doesn't run in parallel
// Used for debugging
//void kernel backprop_step_debug(
//    global float* W,
//    global float* gW,
//    global float* gB,
//    global float* A,
//    global float* prevA,
//    global float* gA,
//    global float* prevgA,
//    const uint cols,
//    const uint p_cols)
//{
//    int id = get_global_id(0);
//    // Everything runs on worker 0
//    if(id != 0) return;
//
//    for(int j = 0; j < cols; j++) {
//
//        float aL = A[j];
//
//        // (aL - y)
//        float diff_ay = gA[j];
//
//        //float relu_p = relu_prime(aL);
//        float relu_p = aL * (1.0 - aL);
//
//        float delta = 2.0 * diff_ay * relu_p;
//
//        gB[j] += delta;
//
//        for(int k = 0; k < p_cols; k++) {
//            float aLprev = prevA[k];
//
//            float weight = W[k*cols + j];
//
//            gW[k*cols + j] += aLprev * delta;
//
//            prevgA[k] += weight * delta;
//        }
//    }
//}
