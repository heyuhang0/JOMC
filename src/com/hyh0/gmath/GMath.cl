// let mr = m1 + m2
kernel void matrixAdd(global const float* m1, global const float* m2, global float* mr) {
    int iGID = get_global_id(0);
    mr[iGID] = m1[iGID] + m2[iGID];
}

// let mr = m1 - m2
kernel void matrixSubtract(global const float* m1, global const float* m2, global float* mr) {
 int iGID = get_global_id(0);
 mr[iGID] = m1[iGID] - m2[iGID];
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

// transpose matrix
kernel void transpose(global const float* matrix, global float* mr, int M, int N) {
    int m = get_global_id(0);
    int n = get_global_id(1);
    
    mr[n * M + m] = matrix[m * N + n];
}

// copy a matrix
kernel void copy(global const float* originalMatrix, global float* newMatrix) {
    int id = get_global_id(0);
    newMatrix[id] = originalMatrix[id];
}

// let mr = k * m
kernel void matrixScalarMultiply(global const float* m, float k, global float* mr) {
    int iGID = get_global_id(0);
    mr[iGID] = m[iGID] * k;
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

// let mr = m1 * m2
// M : number of rows in m1 (must can be divisible by 8)
// N : number of rows in m2 (also number of columns in m1) (must can be divisible by 8)
// P : number of columns in m2
// this function is faster than the one above
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
        
        if(n == 0) {
            for(int m = 0; m < WORK_ITEM_M; m++) {
                for(int n = 0; n < WORK_ITEM_N; n++) {
                    sum[m][n] = data1[m] * data2[n];
                }
            }
        } else {
            for(int m = 0; m < WORK_ITEM_M; m++) {
                for(int n = 0; n < WORK_ITEM_N; n++) {
                    sum[m][n] += data1[m] * data2[n];
                }
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
    resultMatrix[id] = 1.0f / (1.0f + half_exp(-inputMatrix[id]));
}

// compare two matrix
// and save the number of elements that are differnt in result
#define ERROR_ALLOWED 0.0000001f
kernel void compare(global const float* m1, global const float* m2, global int* result) {
    int id = get_global_id(0);
    float error = m1[id] - m2[id];
    if(error > ERROR_ALLOWED || error < -ERROR_ALLOWED)
        atomic_inc(result);
}

kernel void kAbs(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = fabs(inputMatrix[id]);
}

kernel void kAcos(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = acos(inputMatrix[id]);
}

kernel void kAsin(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = asin(inputMatrix[id]);
}

kernel void kAtan(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = atan(inputMatrix[id]);
}

kernel void kCos(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_cos(inputMatrix[id]);
}

kernel void kSin(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_sin(inputMatrix[id]);
}

kernel void kTan(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_tan(inputMatrix[id]);
}

kernel void kCosh(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = cosh(inputMatrix[id]);
}

kernel void kSinh(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = sinh(inputMatrix[id]);
}

kernel void kTanh(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = tanh(inputMatrix[id]);
}

kernel void kLog(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_log(inputMatrix[id]);
}

kernel void kLog2(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_log2(inputMatrix[id]);
}

kernel void kLog10(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_log10(inputMatrix[id]);
}

kernel void kExp(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_log(inputMatrix[id]);
}

kernel void kExp2(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_log2(inputMatrix[id]);
}

kernel void kExp10(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_log10(inputMatrix[id]);
}

kernel void kSqrt(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_sqrt(inputMatrix[id]);
}

kernel void kRsqrt(
        global const float* inputMatrix, 
        global float* resultMatrix) {
    int id = get_global_id(0);
    resultMatrix[id] = half_rsqrt(inputMatrix[id]);
}

kernel void kPow(
        global const float* inputMatrix, 
        global float* resultMatrix,
        float power) {
    int id = get_global_id(0);
    resultMatrix[id] = pow(inputMatrix[id], power);
}

kernel void kPown(
        global const float* inputMatrix, 
        global float* resultMatrix,
        int power) {
    int id = get_global_id(0);
    resultMatrix[id] = pown(inputMatrix[id], power);
}

kernel void kPowr(
        global const float* inputMatrix, 
        global float* resultMatrix,
        float power) {
    int id = get_global_id(0);
    resultMatrix[id] = powr(inputMatrix[id], power);
}