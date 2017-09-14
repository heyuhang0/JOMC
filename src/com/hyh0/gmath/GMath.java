package com.hyh0.gmath;

import java.io.IOException;

import java.nio.IntBuffer;

import com.hyh0.gmath.debug.Tools;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLProgram;

class GMath {

    private CLContext context;
    private CLDevice device;
    private CLCommandQueue queue;
    private CLProgram program;

    private CLKernel kMatrixAdd;
    private CLKernel kMatrixSubtract;
    private CLKernel kRand;
    private CLKernel kMatrixMultiply;
    private CLKernel kMatrixMultiplyN;
    private CLKernel kSigmoid;
    private CLKernel kCompare;
    private CLKernel kScalarMultiply;
    private CLKernel kTranspose;
    private CLKernel kCopy;
    
    /**
     * 完成OpenCl的初始化 (!!用完后需要调用release方法释放资源)
     */
    public GMath() {
        context = CLContext.create();
        Tools.println(context);
        device = context.getMaxFlopsDevice();
        Tools.println(device);
        queue = device.createCommandQueue();
        try {
            program = context.createProgram(GMath.class.getResourceAsStream("GMath.cl")).build();
            kMatrixAdd = program.createCLKernel("matrixAdd");
            kMatrixSubtract = program.createCLKernel("matrixSubtract");
            kRand = program.createCLKernel("rand");
            kMatrixMultiply = program.createCLKernel("matrixMultiply");
            kMatrixMultiplyN = program.createCLKernel("matrixMultiplyN");
            kSigmoid = program.createCLKernel("sigmoid");
            kCompare = program.createCLKernel("compare");
            kScalarMultiply = program.createCLKernel("matrixScalarMultiply");
            kTranspose = program.createCLKernel("transpose");
            kCopy = program.createCLKernel("copy");
        } catch (IOException e) {
            this.release();
            e.printStackTrace();
        } catch (RuntimeException e) {
            this.release();
            throw e;
        }

    }

    /**
     * 转置矩阵
     * @param 原矩阵
     * @param 储存结果的矩阵(不能与原矩阵相同)
     */
    public void transpose(Matrix m, Matrix result) {
        if (m.getRowDimension() != result.getColumnDimension() || m.getColumnDimension() != result.getRowDimension()) {
            throw newIllegalArgumentException("矩阵大小不符合转制条件");
        } else if (m == result) {
            throw newIllegalArgumentException("转置矩阵的原矩阵与结果矩阵不能相同");
        } else {
            kTranspose.setArg(0,  m.getArg());
            kTranspose.setArg(1, result.getArg());
            kTranspose.setArg(2, m.getRowDimension());
            kTranspose.setArg(3, m.getColumnDimension());
            queue.put2DRangeKernel(kTranspose, 0, 0, m.getRowDimension(), m.getColumnDimension(), 0, 0);
        }
    }
    
    public void copy(Matrix originalMatrix, Matrix newMatrix) {
        if (originalMatrix.getRowDimension() != newMatrix.getRowDimension() || originalMatrix.getColumnDimension() != newMatrix.getColumnDimension()) {
            throw newIllegalArgumentException("矩阵大小不符");
        } else {
            kCopy.setArg(0, originalMatrix.getArg());
            kCopy.setArg(1, newMatrix.getArg());
            queue.put1DRangeKernel(kCopy, 0, originalMatrix.getRowDimension() * originalMatrix.getColumnDimension(), 0);
        }
    }
    /**
     * 将两个矩阵相加并将结果保存在第三个矩阵中
     * 
     * @param m1
     *            输入矩阵1
     * @param m2
     *            输入矩阵2
     * @param mr
     *            保存结果的矩阵
     */
    public void add(Matrix m1, Matrix m2, Matrix mr) {
        if (m1.getRowDimension() == m2.getRowDimension() 
                && m1.getRowDimension() == mr.getRowDimension() 
                && m1.getColumnDimension() == m2.getColumnDimension() 
                && m1.getColumnDimension() == mr.getColumnDimension()) {
            kMatrixAdd.setArg(0, m1.getArg());
            kMatrixAdd.setArg(1, m2.getArg());
            kMatrixAdd.setArg(2, mr.getArg());
            queue.put1DRangeKernel(kMatrixAdd, 0, m1.getRowDimension() * m1.getColumnDimension(), 0); // 执行内核
        } else {
            throw newIllegalArgumentException("矩阵的大小不同,无法相加");
        }
    }
    
    /**
     * 将两个矩阵相减并将结果保存在第三个矩阵中
     * 
     * @param m1
     *            输入矩阵1
     * @param m2
     *            输入矩阵2
     * @param mr
     *            保存结果的矩阵
     */
    public void substract(Matrix m1, Matrix m2, Matrix mr) {
        if (m1.getRowDimension() == m2.getRowDimension() 
                && m1.getRowDimension() == mr.getRowDimension() 
                && m1.getColumnDimension() == m2.getColumnDimension() 
                && m1.getColumnDimension() == mr.getColumnDimension()) {
            kMatrixSubtract.setArg(0, m1.getArg());
            kMatrixSubtract.setArg(1, m2.getArg());
            kMatrixSubtract.setArg(2, mr.getArg());
            queue.put1DRangeKernel(kMatrixSubtract, 0, m1.getRowDimension() * m1.getColumnDimension(), 0); // 执行内核
        } else {
            throw newIllegalArgumentException("矩阵的大小不同,无法相减");
        }
    }

    /**
     * 矩阵数乘
     * @param m 被数乘的矩阵
     * @param k 常数项
     * @param result 储存结果的矩阵
     */
    public void multiply(Matrix m, double k, Matrix result) {
        if(m.getRowDimension() == result.getRowDimension() 
                && m.getColumnDimension() == result.getColumnDimension()) {
            kScalarMultiply.setArg(0, m.getArg());
            kScalarMultiply.setArg(1, (float)k);
            kScalarMultiply.setArg(2, result.getArg());
            queue.put1DRangeKernel(kScalarMultiply, 0, m.getRowDimension() * m.getColumnDimension(), 0);
        } else {
            throw newIllegalArgumentException("矩阵的大小不同,无法保存结果");
        }
    }
    /**
     * 用均匀随机数初始化矩阵
     * 
     * @param matrix
     *            输出的矩阵
     * @param lowerLimit
     *            随机数的下限
     * @param upperLimit
     *            随机数的上限
     */
    // TODO 当前只使用了一个粗糙的伪随机算法
    public void fillMatrixRandomly(Matrix matrix, double lowerLimit, double upperLimit) {
        kRand.setArg(0, matrix.getArg());
        kRand.setArg(1, (float) lowerLimit);
        kRand.setArg(2, (float) upperLimit);
        kRand.setArg(3, (int) (Math.random() * 100));
        queue.put1DRangeKernel(kRand, 0, matrix.getRowDimension() * matrix.getColumnDimension(), 0);
    }

    /**
     * 将两个矩阵相乘并将结果保存在第三个矩阵中
     * 
     * @param m1
     *            输入矩阵1
     * @param m2
     *            输入矩阵2
     * @param mr
     *            保存结果的矩阵
     */
    public void multiply(Matrix m1, Matrix m2, Matrix mr) {
        final int MULTIPLY_WORK_ITEM_M = 8;
        final int MULTIPLY_WORK_ITEM_N = 8;
        if (m1.getRowDimension() == mr.getRowDimension() 
                && m1.getColumnDimension() == m2.getRowDimension() 
                && m2.getColumnDimension() == mr.getColumnDimension()) {

            int globalWorkSizeM = m1.getRowDimension() / MULTIPLY_WORK_ITEM_M; // 向下取整
            int globalWorkSizeN = m2.getColumnDimension() / MULTIPLY_WORK_ITEM_N;

            int offsetM = globalWorkSizeM * MULTIPLY_WORK_ITEM_M;
            int offsetN = globalWorkSizeN * MULTIPLY_WORK_ITEM_N;

            int globalWorkSizeReamainM = m1.getRowDimension() - offsetM;
            int globalWorkSizeReamainN = m2.getColumnDimension() - offsetN;

            if (globalWorkSizeM != 0 && globalWorkSizeN != 0) {
                kMatrixMultiplyN.setArg(0, m1.getArg());
                kMatrixMultiplyN.setArg(1, m2.getArg());
                kMatrixMultiplyN.setArg(2, mr.getArg());
                kMatrixMultiplyN.setArg(3, m1.getRowDimension());
                kMatrixMultiplyN.setArg(4, m1.getColumnDimension());
                kMatrixMultiplyN.setArg(5, m2.getRowDimension());
                queue.put2DRangeKernel(kMatrixMultiplyN, 0, 0, m1.getRowDimension() / MULTIPLY_WORK_ITEM_M, m2.getColumnDimension() / MULTIPLY_WORK_ITEM_N,
                        0, 0);
            }

            if (m1.getRowDimension() % MULTIPLY_WORK_ITEM_M != 0) {
                kMatrixMultiply.setArg(0, m1.getArg());
                kMatrixMultiply.setArg(1, m2.getArg());
                kMatrixMultiply.setArg(2, mr.getArg());
                kMatrixMultiply.setArg(3, m1.getRowDimension());
                kMatrixMultiply.setArg(4, m1.getColumnDimension());
                kMatrixMultiply.setArg(5, m2.getColumnDimension());
                queue.put2DRangeKernel(kMatrixMultiply, offsetM, 0, globalWorkSizeReamainM, m2.getColumnDimension(), 0, 0);
            }

            if (m2.getColumnDimension() % MULTIPLY_WORK_ITEM_N != 0 && m1.getRowDimension() > globalWorkSizeReamainM) {
                kMatrixMultiply.setArg(0, m1.getArg());
                kMatrixMultiply.setArg(1, m2.getArg());
                kMatrixMultiply.setArg(2, mr.getArg());
                kMatrixMultiply.setArg(3, m1.getRowDimension());
                kMatrixMultiply.setArg(4, m1.getColumnDimension());
                kMatrixMultiply.setArg(5, m2.getColumnDimension());
                queue.put2DRangeKernel(kMatrixMultiply, 0, offsetN, m1.getRowDimension() - globalWorkSizeReamainM,
                        globalWorkSizeReamainN, 0, 0);
            }

        } else {
            throw newIllegalArgumentException("矩阵的大小不符合相乘的条件");
        }
    }

    /**
     * 将输入矩阵中的值经过sigmoid函数计算后储存在结果矩阵中
     * 
     * @param inputMatrix
     *            输入矩阵
     * @param resultMatrix
     *            结果矩阵
     */
    public void sigmoid(Matrix inputMatrix, Matrix resultMatrix) {
        if (inputMatrix.getRowDimension() == resultMatrix.getRowDimension() && inputMatrix.getRowDimension() == resultMatrix.getRowDimension()) {
            kSigmoid.setArg(0, inputMatrix.getArg());
            kSigmoid.setArg(1, resultMatrix.getArg());

            queue.put1DRangeKernel(kSigmoid, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
        } else {
            throw newIllegalArgumentException("输入矩阵与输出矩阵大小不同");
        }
    }

    private CLBuffer<IntBuffer> isEqualResultBuffer;
    private boolean isEqualResultBufferInited = false;

    /**
     * 比较两个矩阵是否相等
     * 
     * @param m1
     *            矩阵1
     * @param m2
     *            矩阵2
     * @return 如果矩阵相等返回true
     */
    public boolean compare(Matrix m1, Matrix m2) {
        if (m1.getRowDimension() == m2.getRowDimension() && m1.getColumnDimension() == m2.getColumnDimension()) {
            if (!isEqualResultBufferInited) {
                isEqualResultBuffer = context.createIntBuffer(1, CLMemory.Mem.READ_WRITE);
            }
            isEqualResultBuffer.getBuffer().position(0);
            isEqualResultBuffer.getBuffer().put(0);
            isEqualResultBuffer.getBuffer().position(0);
            queue.putWriteBuffer(isEqualResultBuffer, true);
            kCompare.setArg(0, m1.getArg());
            kCompare.setArg(1, m2.getArg());
            kCompare.setArg(2, isEqualResultBuffer);
            queue.put1DRangeKernel(kCompare, 0, m1.getRowDimension() * m2.getColumnDimension(), 0);
            queue.putReadBuffer(isEqualResultBuffer, true);

            if (isEqualResultBuffer.getBuffer().get(0) > 0)
                return false;
            else
                return true;
        } else {
            throw newIllegalArgumentException("矩阵的大小不符,无法比较");
        }
    }

    /**
     * 等待队列中计算全部完成
     */
    public void finish() {
        queue.finish();
    }

    public CLCommandQueue getQueue() {
        return this.queue;
    }
    
    public CLContext getContext() {
        return this.context;
    }
    
    @Override
    protected void finalize() {
        this.release();
    }

    /**
     * 释放 OpenCl 的资源 (必须在程序结束前被调用)
     */
    public void release() {
        context.release();
        Tools.println("context被成功释放");
    }

    /**
     * 创建不合法参数异常，同时释放资源
     * 
     * @param message
     *            包含的信息
     * @return IllegalArgument 异常
     */
    private IllegalArgumentException newIllegalArgumentException(String message) {
        this.release();
        return new IllegalArgumentException(message);
    }
}
