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
    private CLKernel kCompare;
    private CLKernel kScalarMultiply;
    private CLKernel kTranspose;
    private CLKernel kCopy;
    private CLKernel kCopy2D;
    private CLKernel kArrayMultiply;
    private CLKernel kArrayDivide;
    private CLKernel kScalarDivide;

    private CLKernel kAbs;
    private CLKernel kAcos;
    private CLKernel kAsin;
    private CLKernel kAtan;
    private CLKernel kCos;
    private CLKernel kSin;
    private CLKernel kTan;
    private CLKernel kCosh;
    private CLKernel kSinh;
    private CLKernel kTanh;
    private CLKernel kLog;
    private CLKernel kLog2;
    private CLKernel kLog10;
    private CLKernel kExp;
    private CLKernel kExp2;
    private CLKernel kExp10;
    private CLKernel kSqrt;
    private CLKernel kRsqrt;
    private CLKernel kPow; // 以矩阵元素为底，另一个数为指数
    private CLKernel kPow2; // 以另一个数为底，另一个元素为指数
    private CLKernel kPown;
    private CLKernel kSigmoid;

    private final int MULTIPLY_WORK_ITEM_M = 8; // 矩阵乘法每个工作项处理的矩阵行数(需要与cl中的大小对应)
    private final int MULTIPLY_WORK_ITEM_N = 8; // 矩阵乘法每个工作项处理的矩阵列数(需要与cl中的大小对应)
    private int groupSizeForMultiplicationM; // 对于矩阵乘法的最优工作组大小(m方向)
    private int groupSizeForMultiplicationN; // 对于矩阵乘法的最优工作组大小(n方向)

    /**
     * 完成OpenCl的初始化 (!!用完后需要调用release方法释放资源)
     * 
     * @param deviceType
     *            设备种类(CPU/GPU)
     */
    public GMath(CLDevice.Type deviceType) {
        // Tools.setPrint(true); // 打开debug输出
        context = CLContext.create();
        Tools.println(context);
        device = context.getMaxFlopsDevice(deviceType);
        Tools.println(device);
        queue = device.createCommandQueue();
        Tools.println("Preferred Float Vector Width: " + device.getPreferredFloatVectorWidth());
        Tools.println("Max Work Group Size: " + device.getMaxWorkGroupSize());
        Tools.println("The number of CUs: " + device.getMaxComputeUnits());
        try {
            program = context.createProgram(GMath.class.getResourceAsStream("GMath.cl")).build();
            Tools.println(program.getBuildLog());
            kMatrixAdd = program.createCLKernel("matrixAdd");
            kMatrixSubtract = program.createCLKernel("matrixSubtract");
            kRand = program.createCLKernel("rand");
            kMatrixMultiply = program.createCLKernel("matrixMultiply");
            kMatrixMultiplyN = program.createCLKernel("matrixMultiplyN");
            kCompare = program.createCLKernel("compare");
            kScalarMultiply = program.createCLKernel("matrixScalarMultiply");
            kTranspose = program.createCLKernel("transpose");
            kCopy = program.createCLKernel("copy");
            kCopy2D = program.createCLKernel("copy2D");
            kArrayMultiply = program.createCLKernel("arrayMultiply");
            kArrayDivide = program.createCLKernel("arrayDivide");
            kScalarDivide = program.createCLKernel("scalarDivide");

            kSigmoid = program.createCLKernel("sigmoid");
            kAbs = program.createCLKernel("kAbs");
            kAcos = program.createCLKernel("kAcos");
            kAsin = program.createCLKernel("kAsin");
            kAtan = program.createCLKernel("kAtan");
            kCos = program.createCLKernel("kCos");
            kSin = program.createCLKernel("kSin");
            kTan = program.createCLKernel("kTan");
            kCosh = program.createCLKernel("kCosh");
            kSinh = program.createCLKernel("kSinh");
            kTanh = program.createCLKernel("kTanh");
            kLog = program.createCLKernel("kLog");
            kLog2 = program.createCLKernel("kLog2");
            kLog10 = program.createCLKernel("kLog10");
            kExp = program.createCLKernel("kExp");
            kExp2 = program.createCLKernel("kExp2");
            kExp10 = program.createCLKernel("kExp10");
            kSqrt = program.createCLKernel("kSqrt");
            kRsqrt = program.createCLKernel("kRsqrt");
            kPow = program.createCLKernel("kPow");
            kPow2 = program.createCLKernel("kPow2");
            kPown = program.createCLKernel("kPown");
        } catch (IOException e) {
            this.release();
            e.printStackTrace();
        } catch (RuntimeException e) {
            this.release();
            throw e;
        }
        setGroupSizeForMultiplication();
    }

    /*
     * 对CU数因式分解成2个最接近的数作为工作组的M和N的大小
     * 只在 HD5300 与 CPU 上进行过测试，无法保证是否是最佳工作组大小
     */
    private void setGroupSizeForMultiplication() {
        int maxCU = device.getMaxComputeUnits();
        for (groupSizeForMultiplicationM = (int) Math.sqrt(maxCU); maxCU
                % groupSizeForMultiplicationM != 0; groupSizeForMultiplicationM--)
            ;
        groupSizeForMultiplicationN = maxCU / groupSizeForMultiplicationM;

        Tools.println("work group size for multiplication: " + groupSizeForMultiplicationM + "*"
                + groupSizeForMultiplicationN);
    }

    /**
     * 转置矩阵
     * 
     * @param 原矩阵
     * @param 储存结果的矩阵(不能与原矩阵相同)
     */
    public void transpose(Matrix m, Matrix result) {
        if (m.getRowDimension() != result.getColumnDimension() || m.getColumnDimension() != result.getRowDimension()) {
            throw newIllegalArgumentException("矩阵大小不符合转制条件", m, result);
        } else if (m == result) {
            throw newIllegalArgumentException("转置矩阵的原矩阵与结果矩阵不能相同", m, result);
        } else {
            kTranspose.setArg(0, m.getArg());
            kTranspose.setArg(1, result.getArg());
            kTranspose.setArg(2, m.getRowDimension());
            kTranspose.setArg(3, m.getColumnDimension());
            queue.put2DRangeKernel(kTranspose, 0, 0, m.getRowDimension(), m.getColumnDimension(), 0, 0);
        }
    }

    public void copy(Matrix originalMatrix, Matrix newMatrix) {
        checkMatrix(originalMatrix, newMatrix);
        kCopy.setArg(0, originalMatrix.getArg());
        kCopy.setArg(1, newMatrix.getArg());
        queue.put1DRangeKernel(kCopy, 0, originalMatrix.getRowDimension() * originalMatrix.getColumnDimension(), 0);
    }

    /**
     * 把矩阵的一个区域复制到另一个矩阵的一个区域
     * 
     * @param originalMatrix
     *            原矩阵
     * @param startPointMO
     *            原矩阵要复制的区域左上角的行坐标
     * @param startPointNO
     *            原矩阵要复制的区域左上角的列坐标
     * @param newMatrix
     *            新矩阵
     * @param startPointMN
     *            新矩阵要复制的区域左上角的行坐标
     * @param startPointNN
     *            新矩阵要复制的区域左上角的列坐标
     * @param mLength
     *            要复制的行数
     * @param nLength
     *            要复制的列数
     */
    public void copy(Matrix originalMatrix, int startPointMO, int startPointNO, Matrix newMatrix, int startPointMN,
            int startPointNN, int mLength, int nLength) {
        // TODO check arguments
        if (startPointMO + mLength > originalMatrix.getRowDimension()
                || startPointNO + nLength > originalMatrix.getColumnDimension()
                || startPointMN + mLength > newMatrix.getRowDimension()
                || startPointNN + nLength > newMatrix.getColumnDimension() || startPointMO < 0 || startPointNO < 0
                || startPointMN < 0 || startPointNO < 0 || mLength <= 0 || nLength <= 0) {
            this.release();
            String message = "复制区域超出矩阵范围\n";
            message += "原矩阵大小: " + originalMatrix.getRowDimension() + "*" + originalMatrix.getColumnDimension() + "\n";
            message += "复制区域大小: (" + startPointMO + "," + startPointNO + ")(" + startPointMO + mLength + ","
                    + startPointNO + nLength + ")\n";
            message += "现矩阵大小: " + newMatrix.getRowDimension() + "*" + newMatrix.getColumnDimension() + "\n";
            message += "粘贴区域大小: (" + startPointMN + "," + startPointNN + ")(" + startPointMN + mLength + ","
                    + startPointNN + nLength + ")\n";
            throw new IllegalArgumentException(message);
        }
        kCopy2D.setArg(0, originalMatrix.getArg());
        kCopy2D.setArg(1, newMatrix.getArg());
        kCopy2D.setArg(2, originalMatrix.getColumnDimension());
        kCopy2D.setArg(3, newMatrix.getColumnDimension());
        kCopy2D.setArg(4, startPointMN - startPointMO);
        kCopy2D.setArg(5, startPointNN - startPointNO);
        kCopy2D.setArg(6, startPointMO);
        kCopy2D.setArg(7, startPointNO);
        queue.put2DRangeKernel(kCopy2D, 0, 0, mLength, nLength, 0, 0);
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
        checkMatrix(m1, m2);
        checkMatrix(m2, mr);
        kMatrixAdd.setArg(0, m1.getArg());
        kMatrixAdd.setArg(1, m2.getArg());
        kMatrixAdd.setArg(2, mr.getArg());
        queue.put1DRangeKernel(kMatrixAdd, 0, m1.getRowDimension() * m1.getColumnDimension(), 0); // 执行内核
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
        checkMatrix(m1, m2);
        checkMatrix(m2, mr);
        kMatrixSubtract.setArg(0, m1.getArg());
        kMatrixSubtract.setArg(1, m2.getArg());
        kMatrixSubtract.setArg(2, mr.getArg());
        queue.put1DRangeKernel(kMatrixSubtract, 0, m1.getRowDimension() * m1.getColumnDimension(), 0); // 执行内核
    }

    /**
     * 矩阵数乘
     * 
     * @param m
     *            被数乘的矩阵
     * @param k
     *            常数项
     * @param result
     *            储存结果的矩阵
     */
    public void multiply(Matrix m, double k, Matrix result) {
        checkMatrix(m, result);
        kScalarMultiply.setArg(0, m.getArg());
        kScalarMultiply.setArg(1, (float) k);
        kScalarMultiply.setArg(2, result.getArg());
        queue.put1DRangeKernel(kScalarMultiply, 0, m.getRowDimension() * m.getColumnDimension(), 0);
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
        if (m1.getRowDimension() == mr.getRowDimension() && m1.getColumnDimension() == m2.getRowDimension()
                && m2.getColumnDimension() == mr.getColumnDimension()) {

            // 第一轮运算时(8*8为一组计算)时的Work Size大小,向下取整
            int globalWorkSizeM = m1.getRowDimension() / MULTIPLY_WORK_ITEM_M;
            int globalWorkSizeN = m2.getColumnDimension() / MULTIPLY_WORK_ITEM_N;

            // 第二轮运算(计算第一轮的剩余元素时)的起始点
            int offsetM = globalWorkSizeM * MULTIPLY_WORK_ITEM_M;
            int offsetN = globalWorkSizeN * MULTIPLY_WORK_ITEM_N;

            // 第二轮运算(计算第一轮的剩余元素时)的Work Size大小
            int globalWorkSizeReamainM = m1.getRowDimension() - offsetM;
            int globalWorkSizeReamainN = m2.getColumnDimension() - offsetN;

            if (globalWorkSizeM != 0 && globalWorkSizeN != 0) {
                kMatrixMultiplyN.setArg(0, m1.getArg());
                kMatrixMultiplyN.setArg(1, m2.getArg());
                kMatrixMultiplyN.setArg(2, mr.getArg());
                kMatrixMultiplyN.setArg(3, m1.getRowDimension());
                kMatrixMultiplyN.setArg(4, m1.getColumnDimension());
                kMatrixMultiplyN.setArg(5, m2.getColumnDimension());
                kMatrixMultiplyN.setArg(6, globalWorkSizeM);
                kMatrixMultiplyN.setArg(7, globalWorkSizeN);
                queue.put2DRangeKernel(kMatrixMultiplyN, 0, 0,
                        roundUp(groupSizeForMultiplicationM, globalWorkSizeM),
                        roundUp(groupSizeForMultiplicationN, globalWorkSizeN),
                        groupSizeForMultiplicationM,
                        groupSizeForMultiplicationN);
            }
            if (m1.getRowDimension() % MULTIPLY_WORK_ITEM_M != 0) {
                kMatrixMultiply.setArg(0, m1.getArg());
                kMatrixMultiply.setArg(1, m2.getArg());
                kMatrixMultiply.setArg(2, mr.getArg());
                kMatrixMultiply.setArg(3, m1.getRowDimension());
                kMatrixMultiply.setArg(4, m1.getColumnDimension());
                kMatrixMultiply.setArg(5, m2.getColumnDimension());
                queue.put2DRangeKernel(kMatrixMultiply, offsetM, 0, globalWorkSizeReamainM, m2.getColumnDimension(), 0,
                        0);
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
            throw newIllegalArgumentException("矩阵的大小不符合相乘的条件", m1, m2, mr);
        }
    }

    private static int roundUp(int groupSize, int globalSize) {
        if (groupSize <= 0)
            return globalSize;
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

    public void arrayTimes(Matrix m1, Matrix m2, Matrix mr) {
        checkMatrix(m1, m2);
        checkMatrix(m1, mr);
        kArrayMultiply.setArg(0, m1.getArg());
        kArrayMultiply.setArg(1, m2.getArg());
        kArrayMultiply.setArg(2, mr.getArg());
        queue.put1DRangeKernel(kArrayMultiply, 0, m1.getRowDimension() * m1.getColumnDimension(), 0);
    }

    public void arrayDivides(Matrix m1, Matrix m2, Matrix mr) {
        checkMatrix(m1, m2);
        checkMatrix(m1, mr);
        kArrayDivide.setArg(0, m1.getArg());
        kArrayDivide.setArg(1, m2.getArg());
        kArrayDivide.setArg(2, mr.getArg());
        queue.put1DRangeKernel(kArrayDivide, 0, m1.getRowDimension() * m1.getColumnDimension(), 0);
    }

    public void scalarDivides(double k, Matrix m, Matrix mr) {
        checkMatrix(m, mr);
        kScalarDivide.setArg(0, (float) k);
        kScalarDivide.setArg(1, m.getArg());
        kScalarDivide.setArg(2, mr.getArg());
        queue.put1DRangeKernel(kScalarDivide, 0, m.getRowDimension() * m.getColumnDimension(), 0);
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
        checkMatrix(inputMatrix, resultMatrix);
        kSigmoid.setArg(0, inputMatrix.getArg());
        kSigmoid.setArg(1, resultMatrix.getArg());
        queue.put1DRangeKernel(kSigmoid, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
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
     * @param errorAllowed
     *            允许的误差
     * @return 如果矩阵相等返回true
     */
    public boolean compare(Matrix m1, Matrix m2, double errorAllowed) {
        checkMatrix(m1, m2);
        if (!isEqualResultBufferInited) {
            isEqualResultBuffer = context.createIntBuffer(1, CLMemory.Mem.READ_WRITE);
            isEqualResultBufferInited = true;
        }
        isEqualResultBuffer.getBuffer().position(0);
        isEqualResultBuffer.getBuffer().put(0);
        isEqualResultBuffer.getBuffer().position(0);
        queue.putWriteBuffer(isEqualResultBuffer, false);
        kCompare.setArg(0, m1.getArg());
        kCompare.setArg(1, m2.getArg());
        kCompare.setArg(2, isEqualResultBuffer);
        kCompare.setArg(3, (float) errorAllowed);
        queue.put1DRangeKernel(kCompare, 0, m1.getRowDimension() * m2.getColumnDimension(), 0);
        queue.putReadBuffer(isEqualResultBuffer, true);

        if (isEqualResultBuffer.getBuffer().get(0) > 0)
            return false;
        else
            return true;
    }

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
        return compare(m1, m2, 0.000001);
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

    public void abs(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kAbs.setArg(0, inputMatrix.getArg());
        kAbs.setArg(1, resultMatrix.getArg());
        queue.put1DRangeKernel(kAbs, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);

    }

    public void acos(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kAcos.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kAcos, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);

    }

    public void asin(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kAsin.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kAsin, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);

    }

    public void atan(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kAtan.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kAtan, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void cos(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kCos.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kCos, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);

    }

    public void sin(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kSin.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kSin, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);

    }

    public void tan(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kTan.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kTan, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void cosh(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kCosh.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kCosh, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void sinh(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kSinh.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kSinh, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void tanh(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kTanh.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kTanh, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void log(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kLog.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kLog, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void log2(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kLog2.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kLog2, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void log10(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kLog10.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kLog10, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void exp(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kExp.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kExp, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void exp2(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kExp2.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kExp2, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void exp10(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kExp10.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kExp10, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void sqrt(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kSqrt.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kSqrt, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void rsqrt(Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kRsqrt.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        queue.put1DRangeKernel(kRsqrt, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void pow(Matrix inputMatrix, double power, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kPow.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        kPow.setArg(2, (float) power);
        queue.put1DRangeKernel(kPow, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void pow(double power, Matrix inputMatrix, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kPow2.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        kPow2.setArg(2, (float) power);
        queue.put1DRangeKernel(kPow2, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
    }

    public void pow(Matrix inputMatrix, int power, Matrix resultMatrix) {
        checkMatrix(inputMatrix, resultMatrix);
        kPown.setArgs(inputMatrix.getArg(), resultMatrix.getArg());
        kPown.setArg(2, power);
        queue.put1DRangeKernel(kPown, 0, inputMatrix.getRowDimension() * inputMatrix.getColumnDimension(), 0);
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
     * 检查两个矩阵是否大小相等,如果不想等直接抛出异常
     * 
     * @param A
     *            矩阵A
     * @param B
     *            矩阵B
     * @throws IllegalArgumentException
     */
    private void checkMatrix(Matrix A, Matrix B) throws IllegalArgumentException {
        if (A.getRowDimension() != B.getRowDimension() || A.getColumnDimension() != B.getColumnDimension()) {
            String message = "两矩阵大小不相等， 不满足条件\n";
            message += "matrix A: " + A.getRowDimension() + "*" + A.getColumnDimension() + "\n";
            message += "matrix B: " + B.getRowDimension() + "*" + B.getColumnDimension();
            this.release();
            throw new IllegalArgumentException(message);
        }
    }

    /**
     * 创建不合法参数异常，同时释放资源
     * 
     * @param message
     *            包含的信息
     * @return IllegalArgument 异常
     */
    private IllegalArgumentException newIllegalArgumentException(String message, Matrix... matrixs) {
        this.release();
        message += "\n";
        int index = 1;
        for (Matrix e : matrixs) {
            message += "matrix" + index + ": " + e.getRowDimension() + "*" + e.getColumnDimension() + "\n";
            index++;
        }
        return new IllegalArgumentException(message);
    }
}
