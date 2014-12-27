#define TILEX 4

#define TILEX_SHIFT 2
#define TILEY 4
#define TILEY_SHIFT 2

#define NULL 0

__constant float4 ones = (float4) (1.0f);
__constant float4 epsilon = (float4) (1E-30);

// sequence used for getting the memory positions of the float4xfloat4 blocks of data without indexing
__constant int4 normal_seq = (int4) (0, 1, 2, 3);

/*
 *  Returns the index of the element located in (row, col) in a 
 *  row-major memory ordered matrix (Fortran Type)
 */
int4 get_index(int offset, int r, int c, int nr_c, int4 sequence)
{
	const int off = offset + c;
	int4 result = (int4) (r);
	result += sequence;
	result *= nr_c;
	result += off;
    return result;
}

float4 sigmoid(float4 x)
{
    return ones / ( ones + exp( -x ) ); 
}


// softmax has the same derivative. that's why the same function is valid for it
float4 sigmoid_derivative(float4 sigmoid)
{
  return sigmoid*(ones - sigmoid);
}

float4 cross_entropy(float4 t, float4 y)
{
    return ( t * log(y + epsilon) + (ones - t) * log (ones - y + epsilon) );
}

uchar4 dropout_connect_mask(__global uchar * drop_connect_mask, 
                            int4 idxB, 
                            int offsetB, 
                            int global_size0, 
                            int global_id0)
{
    // Index of B of the float4 that we are accessing is equal to the
    // index of bit4 set that has the mask but we need to address bytes,
    // so the byte that we need to access is (idxB >> 1). Inside this byte
    // we have to sets of 4 bits. If (idxB % 2) equals 0 the bit4 set is 
    // the lower one. If equals 1 the bit4 set is the higher one.
    const int minibatchSize = global_size0 << 2;
    const uchar4 bit4Selector = convert_uchar4(idxB % 2);
    const int commonIdx = (global_id0 << 2) * minibatchSize + (offsetB >> 1);
    
    
    // select byte
    const uchar4 val = (uchar4) (drop_connect_mask[commonIdx + (idxB.x >> 1)],
                                 drop_connect_mask[commonIdx + (idxB.y >> 1)],
                                 drop_connect_mask[commonIdx + (idxB.z >> 1)],
                                 drop_connect_mask[commonIdx + (idxB.w >> 1)]);
                                
    // select bit4 set inside byte
    uchar4 mask = bit4Selector?(val >> (uchar4) (4)):(val & (uchar4) (0x0F));

    return mask;
}

/* Matrix A is cached into local memory block */
/* Required global threads = (colsC / 4, rowsC / 4) 
 * Required sizes: rowsC, colsC, rowsA, colsA, rowsB, colsB
 * multiples of 8.
 */

__kernel void matrixMultiplicationSigmoidKernelLocal
                             (__global float4 *matrixA,
                              __global float4 *matrixB,
                              __global float4 *matrixC,
                              __global float4 *bias,
                              int colsA,
                              int offsetA,
                              int offsetB,
                              int offsetC,
                              int offsetBias,
                              __local float4 *blockA,
                              int calcSigmoid,
                              int AInColMajorOrder,
                              int BInColMajorOrder,
                              int sumToMatrixC,
                              float multPrevVal,
                              float multSum,
                              __global uchar *dropconnectB,
                              float dropconnectInferenceProbability)
{
    const int gid0 = get_global_id(0);
    const int gid1 = get_global_id(1);
    const int lid0 = get_local_id(0);
    const int lid1 = get_local_id(1);
    const int lsz0 = get_local_size(0);
    const int lsz1 = get_local_size(1);
    const int gsz0 = get_global_size(0);
    const int gsz1 = get_global_size(1);
    
    
    int4 blockPos = get_index(0, (lid1 << TILEY_SHIFT), lid0, lsz0, normal_seq);

    /* Position of thread will be according to the number of values it writes i.e TILE size */
    
    const int col_C = gid0;
    const int row_C = gid1;
    const int nr_cols_C = gsz0; 
    int4 globalPos = get_index(offsetC, (row_C << TILEY_SHIFT), col_C, nr_cols_C, normal_seq);

    /* Each thread writes 4 float4s */
    float4 sum0 = (float4)(0);
    float4 sum1 = (float4)(0);
    float4 sum2 = (float4)(0);
    float4 sum3 = (float4)(0);

    int temp = colsA / 4;
    
    /* This loop runs for number of blocks of A in horizontal direction */
    for(int i = 0; i < (temp / lsz0); i++)
    {
        /* Calculate global ids of threads from the particular block to load from matrix A depending on i */
        //int globalPosA = offsetA + i * get_local_size(0) + get_local_id(0) + (get_global_id(1) << TILEY_SHIFT) * temp;

        const int col_A = i * lsz0 + lid0;
        const int row_A = gid1;
        const int nr_rows_A = gsz1;
        const int nr_cols_A = temp; 
        
        if(!AInColMajorOrder) {
          int4 globalPosA = get_index(offsetA, (row_A << TILEY_SHIFT), col_A, nr_cols_A, normal_seq);
          /* Load values in blockA from matrixA */
          blockA[blockPos.x] = matrixA[globalPosA.x];
          blockA[blockPos.y] = matrixA[globalPosA.y];
          blockA[blockPos.z] = matrixA[globalPosA.z];
          blockA[blockPos.w] = matrixA[globalPosA.w];
        } else {
          // If A is in column major order not only the index calculation is different but the float4xfloat4 block
          // of data has to be transposed
          int4 globalPosA = get_index(offsetA, (col_A << TILEY_SHIFT), row_A, nr_rows_A, normal_seq);
          // first of all we load the block to private memory
          float4 v1 = matrixA[globalPosA.x];
          float4 v2 = matrixA[globalPosA.y];
          float4 v3 = matrixA[globalPosA.z];
          float4 v4 = matrixA[globalPosA.w];

          // now we transpose it and assign it to the block of memory
          blockA[blockPos.x] = (float4) (v1.x, v2.x, v3.x, v4.x);
          blockA[blockPos.y] = (float4) (v1.y, v2.y, v3.y, v4.y);
          blockA[blockPos.z] = (float4) (v1.z, v2.z, v3.z, v4.z);
          blockA[blockPos.w] = (float4) (v1.w, v2.w, v3.w, v4.w);

        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Calculate global ids of threads from the particular block to load from matrix B depending on i */
        const int col_B = gid0;
        const int row_B = i * lsz0;
        const int nr_rows_B = temp;
        const int nr_cols_B = gsz0; 

        /* This loop runs for number of threads in horizontal direction in the block of A */
        for(int j = 0; j < lsz0 * 4; j=j+4)
        {
            /* Load 4 float4s from blockA : access patters = strided from local memory */
            float4 tempA0 = blockA[(j >> 2) + lid1 * TILEY * lsz0];
            float4 tempA1 = blockA[(j >> 2) + (lid1 * TILEY + 1) * lsz0];
            float4 tempA2 = blockA[(j >> 2) + (lid1 * TILEY + 2) * lsz0];
            float4 tempA3 = blockA[(j >> 2) + (lid1 * TILEY + 3) * lsz0];

            /* Load corresponding values from matrixB, access pattern = linear from global memory */
            float4 tempB0;
            float4 tempB1;
            float4 tempB2;
            float4 tempB3;

            int4 globalPosB;
            if(!BInColMajorOrder) {
              globalPosB = get_index(offsetB, (row_B << TILEY_SHIFT) + j, col_B, nr_cols_B, normal_seq);
              tempB0 = matrixB[globalPosB.x]; //Should be localId.x * (TILEX / 4)
              tempB1 = matrixB[globalPosB.y];
              tempB2 = matrixB[globalPosB.z];
              tempB3 = matrixB[globalPosB.w];
            } else {
              globalPosB = get_index(offsetB, (col_B << TILEY_SHIFT), row_B + (j >> 2), nr_rows_B, normal_seq);

              // load block in private memory
              float4 v1 = matrixB[globalPosB.x];
              float4 v2 = matrixB[globalPosB.y];
              float4 v3 = matrixB[globalPosB.z];
              float4 v4 = matrixB[globalPosB.w];

              // now we transpose it
              tempB0 = (float4) (v1.x, v2.x, v3.x, v4.x);
              tempB1 = (float4) (v1.y, v2.y, v3.y, v4.y);
              tempB2 = (float4) (v1.z, v2.z, v3.z, v4.z);
              tempB3 = (float4) (v1.w, v2.w, v3.w, v4.w);

            }
            
            // Dropout only required when weight matrix is used as matrixB and in row major order
            // In the feedforward calculation
            // that's why the check is only present here.
            if (dropconnectB) {
                const uchar4 mask = dropout_connect_mask(dropconnectB, 
                                            globalPosB, offsetB, gsz0, gid0);
                // bits girados conscientemente para facilitar el cálculo paralelo
                // creo que no afecta pues son bits randoms y se escogen no repetidos
                tempB0 = (mask & ((uchar4) 0x08))?((float4)(0.0f)):tempB0;
                tempB1 = (mask & ((uchar4) 0x04))?((float4)(0.0f)):tempB1;
                tempB2 = (mask & ((uchar4) 0x02))?((float4)(0.0f)):tempB2;
                tempB3 = (mask & ((uchar4) 0x01))?((float4)(0.0f)):tempB3;                
            }


            sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x + tempA0.z * tempB2.x + tempA0.w * tempB3.x;
            sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y + tempA0.z * tempB2.y + tempA0.w * tempB3.y;
            sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z + tempA0.z * tempB2.z + tempA0.w * tempB3.z;
            sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w + tempA0.z * tempB2.w + tempA0.w * tempB3.w;

            sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x + tempA1.z * tempB2.x + tempA1.w * tempB3.x;
            sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y + tempA1.z * tempB2.y + tempA1.w * tempB3.y;
            sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z + tempA1.z * tempB2.z + tempA1.w * tempB3.z;
            sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w + tempA1.z * tempB2.w + tempA1.w * tempB3.w;

            sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x + tempA2.z * tempB2.x + tempA2.w * tempB3.x;
            sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y + tempA2.z * tempB2.y + tempA2.w * tempB3.y;
            sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z + tempA2.z * tempB2.z + tempA2.w * tempB3.z;
            sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w + tempA2.z * tempB2.w + tempA2.w * tempB3.w;

            sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x + tempA3.z * tempB2.x + tempA3.w * tempB3.x;
            sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y + tempA3.z * tempB2.y + tempA3.w * tempB3.y;
            sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z + tempA3.z * tempB2.z + tempA3.w * tempB3.z;
            sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w + tempA3.z * tempB2.w + tempA3.w * tempB3.w;

        }
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // If bias not NULL
    if(bias != NULL) {
        const float4 bias_val = bias[offsetBias + gid0];

        sum0 += bias_val;
        sum1 += bias_val;
        sum2 += bias_val;
        sum3 += bias_val;
    }

    if(dropconnectInferenceProbability < 0.99999) {
    // dropconnectInferenceProbability (must be 1 when no dropconnect inference)
        sum0 *= dropconnectInferenceProbability;
        sum1 *= dropconnectInferenceProbability;
        sum2 *= dropconnectInferenceProbability;
        sum3 *= dropconnectInferenceProbability;
    }
    // Calculate the sigmoid function of the sum
    if(calcSigmoid) {
	sum0 = sigmoid(sum0);
        sum1 = sigmoid(sum1);
        sum2 = sigmoid(sum2);
        sum3 = sigmoid(sum3);
    }

    // end of calculation of sigmoid function
    
    /* Write 16 values to matrixC */
    if(sumToMatrixC) {
        const float4 a = matrixC[globalPos.x] * multPrevVal;
        const float4 b = matrixC[globalPos.y] * multPrevVal;
        const float4 c = matrixC[globalPos.z] * multPrevVal;
        const float4 d = matrixC[globalPos.w] * multPrevVal;  

        matrixC[globalPos.x] = a + multSum*sum0;
        matrixC[globalPos.y] = b + multSum*sum1;
        matrixC[globalPos.z] = c + multSum*sum2;
        matrixC[globalPos.w] = d + multSum*sum3;    
    } else {
        matrixC[globalPos.x] = sum0;
        matrixC[globalPos.y] = sum1;
        matrixC[globalPos.z] = sum2;
        matrixC[globalPos.w] = sum3;    
    }
}

/* Substracts element by element. NDRange of one dimension. 
 * Take care that every element is a float4 element.
 * The dimension should be the total number of elements divided by 4
 * This function is used to calculate the deltas of the output layer.
 */
__kernel void elementWiseSubstractKernel(__global float4 *A,
                                         __global float4 *B,
                                         __global float4* R,
                                         int offset_A,
                                         int offset_B,
                                         int offset_R)
{
    const int i = get_global_id(0);
    
    const float4 a = A[offset_A + i];
    const float4 b = B[offset_B + i];
    
    R[offset_R + i] =  a - b;
}

/* Adds element by element. NDRange of one dimension. 
 * Take care that every element is a float4 element.
 * The dimension should be the total number of elements divided by 4
 * This function is used to calculate the deltas of the output layer.
 */
__kernel void elementWiseSumKernel(__global float4* A,
                                   __global float4* B,
                                   __global float4* R,
                                   int offset_A,
                                   int offset_B,
                                   int offset_R,
                                   float mult_A,
                                   float mult_B)
{
    const int i = get_global_id(0);
    
    const float4 a = mult_A*A[offset_A + i];
    const float4 b = mult_B*B[offset_B + i];
    
    R[offset_R + i] =  a + b;
}


__kernel void elementWiseMultiplicationBySigmoidDerivativeKernel(
                                         __global float4 *del,
                                         __global float4 *act,
                                         int offset_del,
                                         int offset_act)
{
    int i = get_global_id(0);

    float4 a = sigmoid_derivative(act[offset_act + i]);
    
    del[offset_del + i] *= a;
}


__kernel void crossEntropyKernelLocal(__global float4* t, 
                                      __global float4* y, 
                                      __global float4* output, 
                                      __local float4* sdata,
                                      int offset_y)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    
    float4 y1 = y[offset_y + stride];
    float4 t1 = t[stride];
    float4 i1 = cross_entropy(t1, y1);
    
    float4 y2 = y[offset_y + stride + 1];
    float4 t2 = t[stride + 1];
    float4 i2 = cross_entropy(t2, y2);
    
    sdata[tid] = i1 + i2;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // do reduction in shared mem
    for(unsigned int s = localSize >> 1; s > 0; s >>= 1) 
    {
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) output[bid] = sdata[0];	
}

__kernel void level2RegularizationKernelLocal(__global float4* W, 
                                              __global float4* O, 
                                              __local float4* sdata)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    
    float4 w1 = W[stride];
    float4 i1 = w1*w1;
    
    float4 w2 = W[stride + 1];
    float4 i2 = w2*w2;
    
    sdata[tid] = i1 + i2;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // do reduction in shared mem
    for(unsigned int s = localSize >> 1; s > 0; s >>= 1) 
    {
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) O[bid] = sdata[0];	
}


// Al finalizar la función se obtiene un vector de output de tamaño igual al número de grupos
// que hay que sumar, obteniendo el resultado final

/*
 * Calculates a softmax of local_size float4's => local_size*4 float elements 
 * Required local_size = number of output elements of the softmax to calculate divided by 4
 * Required global size = all the elements / 4 (floats4)
 */
__kernel void softmaxKernelLocal(__global float4* z, 
                                 __local float4* sdata,
                                 int offset_z)
{
    // load shared mem
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    
    unsigned int idx = offset_z + gid;
    
    sdata[lid] = exp(z[idx]);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // calculate the sum of all the elements
    float4 sum = (float4) (0.0f);
    for(int i = 0; i < localSize; i++) {
        sum += sdata[i];
    }
    
    float total = sum.x + sum.y + sum.z + sum.w;
    
    z[idx] = sdata[lid]/total;
}

/* 
 *  1 dimensional NDRange = number of columns of floats / 4 
 *  Sums the values of all the rows
 */
__kernel void rowSumKernel(__global float4 * matrixA,
                           __global float4 *bias_inc,
                           int nrRowsA,
                           float multExisting,
                           float multNew)
{
    const int gid = get_global_id(0);
    const int gsz = get_global_size(0);

    float4 result = (float4) (0.0f);
    for(int i = 0; i < nrRowsA; i++) {
        const int idx = i*gsz + gid;
        result += matrixA[idx];
    }

    const float4 a = multExisting*bias_inc[gid];
    const float4 b = multNew*result;
    
    bias_inc[gid] = a + b;
}
