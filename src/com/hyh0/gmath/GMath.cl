// add m1 with m2 and save the result in mr
kernel void matrixAdd(global const float* m1, global const float* m2, global float* mr) {
    int iGID = get_global_id(0);
    mr[iGID] = m1[iGID] + m2[iGID];
}

// fill matrix with random number (now it is a test code for temporary use)
kernel void rand(global float* matrix, float lowerLimit, float upperLimit, int seed) {
    int iGID = get_global_id(0);
    seed = (seed * iGID) % 10000;
    int times = (66941 * seed + 92655) % 10;
    for(int i = 0; i < times; i++) {
        seed = (66941 * seed + 92655) % 10000;
    }
    matrix[iGID] = (upperLimit - lowerLimit)/10000*seed + lowerLimit;
}

// let mr = m1 * m2
// M : number of rows in m1
// N : number of rows in m2 (also number of columns in m1)
// P : number of columns in m2
kernel void matrixMultiply(
        global const float* m1,
        global const float* m2,
        global float* mr,
        int M, int N, int P) {
    
    int mID = get_global_id(0);
    int nID = get_global_id(1);
    
    float sum = 0;
    
    int m1StartPoint = mID * N;
    for(int i = 0; i < N; i++) {
        sum += m1[m1StartPoint + i] * m2[nID + i * P];
    }
    mr[mID * P + nID] = sum;
}

// use sigmoid function to compute every element in inputMatrix
// and save the result in result matrix
kernel void sigmoid(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = 1.0 / (1.0 + exp(-inputMatrix[id]));
}

kernel void compare(global const float* m1, global const float* m2, global int* result) {
    int id = get_global_id(0);
    if(m1[id] != m2[id])
        atomic_inc(result);
}