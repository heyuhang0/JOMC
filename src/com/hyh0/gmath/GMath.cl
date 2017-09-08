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

kernel void matrixMultiply2(
        global const float* m1,
        global const float* m2,
        global float* mr,
        int M, int N, int P) {
    
    int mID = get_global_id(0) * 2;
    int nID = get_global_id(1) * 2;
    
    float sum[2][2];
    float data[2][2];
    
    for(int i = 0; i < N; i++) {
        data[0][0] = m1[mID * N + i];
        data[0][1] = m1[mID * N + i + N];
        data[1][0] = m2[nID + i * P];
        data[1][1] = m2[nID + i * P + 1];

        sum[0][0] += data[0][0] * data[1][0];
        sum[0][1] += data[0][0] * data[1][1];
        sum[1][0] += data[0][1] * data[1][0];
        sum[1][1] += data[0][1] * data[1][1];
    }
    mr[mID * P + nID] = sum[0][0];
    mr[mID * P + nID + 1] = sum[0][1];
    mr[mID * P + nID + P] = sum[1][0];
    mr[mID * P + nID + 1 + P] = sum[1][1];
}

#define WORK_ITEM_M 8
#define WORK_ITEM_N 8
kernel void matrixMultiplyN(
        global const float* m1,
        global const float* m2,
        global float* mr,
        int M, int N, int P) {
    int mID = get_global_id(0) * WORK_ITEM_M;
    int nID = get_global_id(1) * WORK_ITEM_N;
    
    float sum[WORK_ITEM_M][WORK_ITEM_N];
    float data1[WORK_ITEM_M];
    float data2[WORK_ITEM_N];
    
    for(int n = 0; n < N; n++) {
        for(int i = 0; i < WORK_ITEM_M; i++) {
            data1[i] = m1[(mID + i) * N + n];
        }
        for(int i = 0; i < WORK_ITEM_N; i++) {
            data2[i] = m2[nID + n * P + i];
        }
        
        for(int m = 0; m < WORK_ITEM_M; m++) {
            for(int n = 0; n < WORK_ITEM_N; n++) {
                sum[m][n] += data1[m] * data2[n];
            }
        }
    }
    for(int m = 0; m < WORK_ITEM_M; m++) {
        for(int n = 0; n < WORK_ITEM_N; n++) {
            mr[(mID + m) * P + nID + n] = sum[m][n];
        }
    }
}

// use sigmoid function to compute every element in inputMatrix
// and save the result in result matrix
kernel void sigmoid(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = 1.0 / (1.0 + exp(-inputMatrix[id]));
}

#define ERROR_ALLOWED 0.0000001
kernel void compare(global const float* m1, global const float* m2, global int* result) {
    int id = get_global_id(0);
    float error = m1[id] - m2[id];
    if(error > ERROR_ALLOWED || error < -ERROR_ALLOWED)
        atomic_inc(result);
}
