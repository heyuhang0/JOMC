package com.hyh0.gmath;

import java.io.IOException;
import java.nio.channels.ShutdownChannelGroupException;

import com.hyh0.gmath.debug.Tools;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

public class GMath {

    private CLContext context;
    private CLDevice device;
    private CLCommandQueue queue;
    private CLProgram program;

    private CLKernel kMatrixAdd;
    private CLKernel kRand;
    private CLKernel kMatrixMultiply;
    private CLKernel kSigmoid;
    
    public GMath() {
        context = CLContext.create();
        Tools.println(context);
        device = context.getMaxFlopsDevice();
        Tools.println(device);
        queue = device.createCommandQueue();
        try {
            program = context.createProgram(GMath.class.getResourceAsStream("GMath.cl")).build();
            kMatrixAdd = program.createCLKernel("matrixAdd");
            kRand = program.createCLKernel("rand");
            kMatrixMultiply = program.createCLKernel("matrixMultiply");
            kSigmoid = program.createCLKernel("sigmoid");
        } catch (IOException e) {
            this.release();
            e.printStackTrace();
        } catch (RuntimeException e) {
            this.release();
            throw e;
        }

    }

    /**
     * 创建一个矩阵对象
     * 
     * @param m
     *            矩阵的行数
     * @param n
     *            矩阵的列数
     * @return 一个GMatrix的矩阵对象
     */
    public GMatrix newGMatrix(int m, int n) {
        return new GMatrix(context, queue, m, n);
    }

    /**
     * 通过一个二维数组创建
     * 
     * @param data
     * @return
     */
    public GMatrix newGMatrix(double[][] data) {
        return new GMatrix(context, queue, data);
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
    public void add(GMatrix m1, GMatrix m2, GMatrix mr) {
        if (m1.M == m2.M && m1.M == mr.M && m1.N == m2.N && m1.N == mr.N) {
            kMatrixAdd.setArg(0, m1.getArg());
            kMatrixAdd.setArg(1, m2.getArg());
            kMatrixAdd.setArg(2, mr.getArg());
            queue.put1DRangeKernel(kMatrixAdd, 0, m1.M * m1.N, 0); // 执行内核
        } else {
            throw newIllegalArgumentException("矩阵的大小不同,无法相加");
        }
    }

    public void fillMatrixRandomly(GMatrix matrix, double lowerLimit, double upperLimit) {
        kRand.setArg(0, matrix.getArg());
        kRand.setArg(1, (float) lowerLimit);
        kRand.setArg(2, (float) upperLimit);
        kRand.setArg(3, (int) (Math.random() * 100));
        queue.put1DRangeKernel(kRand, 0, matrix.M * matrix.N, 0);
    }

    public void multiply(GMatrix m1, GMatrix m2, GMatrix mr) {
        if (m1.M == mr.M
            && m1.N == m2.M
            && m2.N == mr.N) {
            
            kMatrixMultiply.setArg(0, m1.getArg());
            kMatrixMultiply.setArg(1, m2.getArg());
            kMatrixMultiply.setArg(2, mr.getArg());
            kMatrixMultiply.setArg(3, m1.M);
            kMatrixMultiply.setArg(4, m1.N);
            kMatrixMultiply.setArg(5, m2.N);
            queue.put2DRangeKernel(kMatrixMultiply, 0, 0, m1.M, m2.N, 0, 0);
        } else {
            throw newIllegalArgumentException("矩阵的大小不符合相乘的条件");
        }
    }
    
    public void sigmoid(GMatrix inputMatrix, GMatrix resultMatrix) {
        if (inputMatrix.M == resultMatrix.M
                && inputMatrix.M == resultMatrix.M) {
            kSigmoid.setArg(0, inputMatrix.getArg());
            kSigmoid.setArg(1, resultMatrix.getArg());
            
            queue.put1DRangeKernel(kSigmoid, 0, inputMatrix.M * inputMatrix.N, 0);
        } else {
            throw newIllegalArgumentException("输入矩阵与输出矩阵大小不同");
        }
    }

    public void finish() {
        queue.finish();
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
     * @return IllegalArgument异常
     */
    private IllegalArgumentException newIllegalArgumentException(String message) {
        this.release();
        return new IllegalArgumentException(message);
    }
}
