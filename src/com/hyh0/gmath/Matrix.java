package com.hyh0.gmath;

import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.io.StreamTokenizer;
import java.nio.FloatBuffer;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.NumberFormat;
import java.util.Locale;

import com.hyh0.gmath.debug.Tools;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;

public class Matrix implements Cloneable {

    private int M;
    private int N;
    private CLBuffer<FloatBuffer> matrixBuffer;

    private static CLCommandQueue queue;
    private static CLContext context;
    private static GMath gMath;
    private static boolean inited = false;

    /*
     * =================================================================
     * >>>>>>>>>>>>>>>>>>>>>>>>>>>> 构造器 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
     * =================================================================
     */

    /**
     * 初始化OpenCl(使用浮点计算能力最强的设备)
     */
    public static void init() {
        init(DeviceType.DEFAULT);
    }
    
    /**
     * 初始化OpenCl
     * @param deviceType 指定设备种类(CLDevice.Type.CPU/GPU)
     */
    public static void init(DeviceType deviceType) {
        if (inited) {
            return;
        } else {
            gMath = new GMath(deviceType.TYPE);
            queue = gMath.getQueue();
            context = gMath.getContext();
            MatrixMath.init(gMath);
            inited = true;
        }
    }

    /**
     * 创建一个新矩阵,元素全部初始化为0
     * 
     * @param m
     *            矩阵的行数
     * @param n
     *            矩阵的列数
     */
    public Matrix(int m, int n) {
        if (!inited) {
            Matrix.init();
        }
        this.matrixBuffer = context.createFloatBuffer(m * n, READ_WRITE);
        this.M = m;
        this.N = n;
        syncToDevice();
    }

    /**
     * 创建一个新矩阵,元素全部初始化为s
     * 
     * @param m
     *            矩阵的行数
     * @param n
     *            矩阵的列数
     * @param s
     *            矩阵元素的初始值
     */
    public Matrix(int m, int n, double s) {
        this(m, n);
        double[][] data = new double[m][n];
        for (int mm = 0; mm < m; mm++) {
            for (int nn = 0; nn < n; nn++) {
                data[mm][nn] = s;
            }
        }
        this.set(data);
    }

    /**
     * 创建一个与二维数组对应的新矩阵
     * 
     * @param data
     *            储存数据的二维数组
     */
    public Matrix(double[][] data) {
        if (!inited) {
            Matrix.init();
        }
        int m = data.length;
        int n = data[0].length;
        for (int i = 0; i < m; i++) {
            if (data[i].length != n)
                throw newIllegalArgumentException("二维数组的每一行长度应该相等");
        }
        this.matrixBuffer = context.createFloatBuffer(m * n, READ_WRITE);
        this.M = m;
        this.N = n;
        this.syncToDevice();
        this.set(data);
    }

    /**
     * 创建一个 m*n 的单位矩阵(对角线填充1， 其余为0)
     * 
     * @param m
     *            矩阵行数
     * @param n
     *            矩阵列数
     * @return 新建的矩阵
     */
    public static Matrix identity(int m, int n) {
        double data[][] = new double[m][n];
        for (int i = 0; i < n && i < m; i++) {
            data[i][i] = 1;
        }
        return new Matrix(data);
    }

    /**
     * 创建一个 n*n 的单位矩阵
     * 
     * @param n
     *            方阵的阶数
     * @return 新建的矩阵
     */
    public static Matrix identity(int n) {
        return identity(n, n);
    }

    /**
     * 创建一个填充随机数的 m*n 的矩阵
     * 
     * @param m
     *            矩阵行数
     * @param n
     *            矩阵列数
     * @return 新建的矩阵
     */
    public static Matrix random(int m, int n) {
        Matrix matrix = new Matrix(m, n);
        matrix.randomize();
        return matrix;
    }

    /**
     * 创建一个填充随机数的 m*n 的矩阵
     * 
     * @param m
     *            矩阵行数
     * @param n
     *            矩阵列数
     * @param lowerLimit
     *            随机数下限
     * @param upperLimit
     *            随机数上限
     * @return 新建的矩阵
     */
    public static Matrix random(int m, int n, double lowerLimit, double upperLimit) {
        Matrix matrix = new Matrix(m, n);
        matrix.randomize(lowerLimit, upperLimit);
        return matrix;
    }

    /**
     * 用二维数组创建一个矩阵
     * 
     * @param data
     *            二维的double数组
     * @return 新建的矩阵
     */
    public static Matrix constructWithCopy(double[][] data) {
        return new Matrix(data);
    }

    /*
     * =================================================================
     * >>>>>>>>>>>>>>>>>>>>>>>>>> 矩阵操作 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
     * =================================================================
     */
    /**
     * 用随机数初始化矩阵
     * 
     * @param lowerLimit
     *            随机数下限
     * @param upperLimit
     *            随机数上限
     */
    public void randomize(double lowerLimit, double upperLimit) {
        gMath.fillMatrixRandomly(this, lowerLimit, upperLimit);
    }

    /**
     * 用随机数初始化矩阵(-1到1的均匀随机数)
     */
    public void randomize() {
        this.randomize(-1, 1);
    }

    /**
     * result = this + B 将当前矩阵加上另一个矩阵的结果保存在result中
     * 
     * @param B
     *            与当前矩阵相加的矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public Matrix plus(Matrix B, Matrix result) {
        gMath.add(this, B, result);
        return result;
    }

    /**
     * this += B 将当前矩阵加上另一个矩阵的结果保存在当前矩阵中
     * 
     * @param B
     *            与当前矩阵相加的矩阵
     * @return 当前矩阵
     */
    public Matrix plusEquals(Matrix B) {
        return this.plus(B, this);
    }

    /**
     * result = this - B 将当前矩阵加上减去另一个矩阵的结果保存在result中
     * 
     * @param B
     *            与当前矩阵相减的矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public Matrix minus(Matrix B, Matrix result) {
        gMath.substract(this, B, result);
        return result;
    }

    /**
     * this -= B 将当前矩阵减去另一个矩阵的结果保存在当前矩阵中
     * 
     * @param B
     *            与当前矩阵相减的矩阵
     * @return 当前矩阵
     */
    public Matrix minusEquals(Matrix B) {
        return this.minus(B, this);
    }

    /**
     * result = this * B 当前矩阵乘上另一个矩阵
     * 
     * @param B
     *            与当前矩阵相乘的矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public Matrix times(Matrix B, Matrix result) {
        gMath.multiply(this, B, result);
        return result;
    }

    /**
     * result = k * this 将当前矩阵乘上一个常数
     * 
     * @param k
     *            与矩阵相乘的常数
     * @param result
     *            储存结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public Matrix times(double k, Matrix result) {
        gMath.multiply(this, k, result);
        return result;
    }

    /**
     * this = k * this 将当前矩阵乘上一个常数
     * 
     * @param k
     *            与矩阵相乘的常数
     * @return 当前矩阵
     */
    public Matrix timesEquals(double k) {
        return this.times(k, this);
    }

    /**
     * result = B . / this 使另一个矩阵被此矩阵点除
     * 
     * @param B
     *            另一个矩阵
     * @param result
     *            保存结果的矩阵
     * @return 保存结果的矩阵
     */
    public Matrix arrayLeftDivide(Matrix B, Matrix result) {
        B.arrayDivides(this, result);
        return result;
    }

    /**
     * this = B . / this 使另一个矩阵被此矩阵点除,并把结果储存在此矩阵
     * 
     * @param B
     *            另一个矩阵
     * @return 当前矩阵
     */
    public Matrix arrayLeftDivideEquals(Matrix B) {
        return arrayLeftDivide(B, this);
    }

    /**
     * result = this . / B 此矩阵点除另一个矩阵
     * 
     * @param B
     *            另一个矩阵
     * @param result
     *            保存结果的矩阵
     * @return 保存结果的矩阵
     */
    public Matrix arrayDivides(Matrix B, Matrix result) {
        gMath.arrayDivides(this, B, result);
        return result;
    }

    /**
     * this = this . / B 此矩阵点除等于另一个矩阵
     * 
     * @param B
     *            另一个矩阵
     * @return 当前矩阵
     */
    public Matrix arrayDividesEquals(Matrix B) {
        return arrayDivides(B, this);
    }

    /**
     * result = this . / B 此矩阵点除另一个矩阵
     * 
     * @param B
     *            另一个矩阵
     * @param result
     *            保存结果的矩阵
     * @return 保存结果的矩阵
     * @deprecated 只是为了兼容JAMA,arrayDivides效果相同
     */
    public Matrix arrayRightDivide(Matrix B, Matrix result) {
        return arrayDivides(B, result);
    }

    /**
     * this = this . / B 此矩阵点除等于另一个矩阵
     * 
     * @param B
     *            另一个矩阵
     * @return 当前矩阵
     * @deprecated 只是为了兼容JAMA,arrayDividesEquals效果相同
     */
    public Matrix arrayRightDivideEquals(Matrix B) {
        return arrayDividesEquals(B);
    }

    /**
     * 矩阵点乘 result = B . * this
     * 
     * @param B
     *            另一个矩阵
     * @param result
     *            保存结果的矩阵
     * @return 保存结果的矩阵
     */
    public Matrix arrayTimes(Matrix B, Matrix result) {
        gMath.arrayTimes(this, B, result);
        return result;
    }

    /**
     * 矩阵点乘等于 this = B . * this
     * 
     * @param B
     *            另一个矩阵
     * @return 当前矩阵
     */
    public Matrix arrayTimesEquals(Matrix B) {
        return arrayTimes(B, this);
    }

    /**
     * 矩阵的每一个元素除一个常数 result = this/s
     * 
     * @param s
     *            除数
     * @param result
     *            保存结果的矩阵
     * @return 保存结果的矩阵
     */
    public Matrix divides(double s, Matrix result) {
        result = times(1 / s, result);
        return result;
    }

    /**
     * 矩阵的每一个元素除等于一个常数 result = this/s
     * 
     * @param s
     *            除数
     * @return 保存结果的矩阵
     */
    public Matrix dividesEquals(double s) {
        return divides(s, this);
    }

    /**
     * 让矩阵的每个元素都被常数除 result = k . / this
     * 
     * @param k
     *            被除数
     * @param result
     *            保存结果的矩阵
     * @return 保存结果的矩阵
     */
    public Matrix leftDivide(double k, Matrix result) {
        gMath.scalarDivides(k, this, result);
        return result;
    }

    /**
     * 让矩阵的每个元素都被常数除,且在原矩阵中保存运算结果 this = s . / this
     * 
     * @param s
     *            被除数
     * @return 当前矩阵
     */
    public Matrix leftDivideEquals(double s) {
        return leftDivide(s, this);
    }

    /**
     * 将矩阵的转置矩阵储存在新矩阵中
     * 
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public Matrix transpose(Matrix result) {
        gMath.transpose(this, result);
        return result;
    }

    /**
     * 将矩阵复制到新的矩阵中
     * 
     * @param newMatrix
     *            新的矩阵
     * @return 保存运算结果的矩阵
     */
    public Matrix copyTo(Matrix newMatrix) {
        gMath.copy(this, newMatrix);
        return newMatrix;
    }

    /**
     * 把矩阵的一个区域的数据复制到另一个矩阵的一个区域
     * 
     * @param mStart
     *            数据复制的起始行
     * @param nStart
     *            数据复制的起始列
     * @param newMatrix
     *            目标矩阵
     * @param mStartOfNew
     *            数据在目标矩阵中的起始行
     * @param nStartOfNew
     *            数据在目标矩阵中的起始列
     * @param numOfRows
     *            要复制的行数
     * @param numOFColumns
     *            要复制的列数
     * @return 目标矩阵
     */
    public Matrix copyTo(int mStart, int nStart, Matrix newMatrix, int mStartOfNew, int nStartOfNew, int numOfRows,
            int numOFColumns) {
        gMath.copy(this, mStart, nStart, newMatrix, mStartOfNew, nStartOfNew, numOfRows, numOFColumns);
        return newMatrix;
    }

    /**
     * 获得当前矩阵的拷贝(deep clone)
     * 
     * @return 复制产生的新矩阵
     */
    public Matrix copy() {
        double data[][] = this.getArrayCopy();
        return new Matrix(data);
    }

    /**
     * Clone the Matrix object.
     * 
     * @return 当前矩阵的拷贝
     */
    @Override
    public Object clone() {
        return this.copy();
    }

    /**
     * 比较两个矩阵是否相等
     * 
     * @param another
     *            与之比较的矩阵
     * @return 如果相等即为true,反之为false
     */
    public boolean isEqualTo(Matrix another) {
        return gMath.compare(this, another);
    }

    /**
     * 设置矩阵的元素,并同步到显存
     * 
     * @param m
     *            第m行
     * @param n
     *            第n列
     * @param data
     *            要修改成的数据
     */
    public void set(int m, int n, double data) {
        if (m >= this.M || n >= this.N)
            throw newIllegalArgumentException("超出矩阵范围");

        int targetPosition = m * this.N + n;
        FloatBuffer buffer = matrixBuffer.getBuffer();
        buffer.position(targetPosition);
        buffer.put((float) data);
        this.syncToDevice();
    }

    /**
     * 设置矩阵的元素,并同步到显存
     * 
     * @param data
     *            对应的二维数组
     */
    public void set(double[][] data) {
        if (data.length != this.M || data[0].length != this.N)
            throw newIllegalArgumentException("数组和矩阵不符");
        FloatBuffer buffer = matrixBuffer.getBuffer();
        for (double[] vs : data) {
            for (double v : vs) {
                buffer.put((float) v);
            }
        }
        this.syncToDevice();
    }

    /**
     * 获取矩阵中的某个元素
     * 
     * @param m
     *            第M行
     * @param n
     *            第N列
     * @return 该位置上的数值
     */
    public double get(int m, int n) {
        if (m >= this.M || n >= this.N)
            throw newIllegalArgumentException("超出矩阵范围");
        this.syncFromDevice();
        int targetPosition = m * this.N + n;
        return matrixBuffer.getBuffer().get(targetPosition);
    }

    /**
     * 将矩阵的数据转换为二维数组
     * 
     * @return 与矩阵大小对应的二维数组
     */
    public double[][] getArrayCopy() {
        this.syncFromDevice();
        FloatBuffer buffer = matrixBuffer.getBuffer();
        double[][] result = new double[M][N];
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                result[m][n] = buffer.get();
            }
        }
        return result;
    }

    /**
     * 将矩阵的数据转换为二维数组
     * 
     * @return 与矩阵大小对应的二维数组
     * @deprecated 只是为了兼容JAMA，这里得到的数组只是个数据拷贝
     */
    public double[][] getArray() {
        return getArrayCopy();
    }

    /**
     * 获取矩阵行数
     * 
     * @return 矩阵的行数
     */
    public int getRowDimension() {
        return M;
    }

    /**
     * Make a one-dimensional row packed copy of the internal array.
     * 
     * @return Matrix elements packed in a one-dimensional array by rows.
     */
    public double[] getRowPackedCopy() {
        double[][] A = getArrayCopy();
        double[] vals = new double[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                vals[i * N + j] = A[i][j];
            }
        }
        return vals;
    }

    /**
     * 获取矩阵列数
     * 
     * @return 矩阵的列数
     */
    public int getColumnDimension() {
        return N;
    }

    /**
     * Make a one-dimensional column packed copy of the internal array.
     * 
     * @return Matrix elements packed in a one-dimensional array by columns.
     */
    public double[] getColumnPackedCopy() {
        double[][] A = getArrayCopy();
        double[] vals = new double[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                vals[i + j * M] = A[i][j];
            }
        }
        return vals;
    }

    /**
     * Get a submatrix.
     * 
     * @param i0
     *            Initial row index
     * @param i1
     *            Final row index
     * @param j0
     *            Initial column index
     * @param j1
     *            Final column index
     * @return A(i0:i1,j0:j1)
     */

    public Matrix getMatrix(int i0, int i1, int j0, int j1) {
        Matrix B = new Matrix(i1 - i0 + 1, j1 - j0 + 1);
        this.copyTo(i0, j0, B, 0, 0, i1 - i0 + 1, j1 - j0 + 1);
        return B;
    }

    /**
     * Get a submatrix.(There is no optimization for openCl)
     * 
     * @param r
     *            Array of row indices.
     * @param c
     *            Array of column indices.
     * @return A(r(:),c(:))
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */

    public Matrix getMatrix(int[] r, int[] c) {
        double[][] A = this.getArrayCopy();
        double[][] B = new double[r.length][c.length];
        try {
            for (int i = 0; i < r.length; i++) {
                for (int j = 0; j < c.length; j++) {
                    B[i][j] = A[r[i]][c[j]];
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
        return new Matrix(B);
    }

    /**
     * Get a submatrix.(There is no optimization for openCl)
     * 
     * @param i0
     *            Initial row index
     * @param i1
     *            Final row index
     * @param c
     *            Array of column indices.
     * @return A(i0:i1,c(:))
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */

    public Matrix getMatrix(int i0, int i1, int[] c) {
        double[][] A = this.getArrayCopy();
        double[][] B = new double[i1 - i0 + 1][c.length];
        try {
            for (int i = i0; i <= i1; i++) {
                for (int j = 0; j < c.length; j++) {
                    B[i - i0][j] = A[i][c[j]];
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
        return new Matrix(B);
    }

    /**
     * Get a submatrix.(There is no optimization for openCl)
     * 
     * @param r
     *            Array of row indices.
     * @param j0
     *            Initial column index
     * @param j1
     *            Final column index
     * @return A(r(:),j0:j1)
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */

    public Matrix getMatrix(int[] r, int j0, int j1) {
        double[][] A = this.getArrayCopy();
        double[][] B = new double[r.length][j1 - j0 + 1];
        try {
            for (int i = 0; i < r.length; i++) {
                for (int j = j0; j <= j1; j++) {
                    B[i][j - j0] = A[r[i]][j];
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
        return new Matrix(B);
    }

    /**
     * Set a submatrix.
     * 
     * @param i0
     *            Initial row index
     * @param i1
     *            Final row index
     * @param j0
     *            Initial column index
     * @param j1
     *            Final column index
     * @param X
     *            A(i0:i1,j0:j1)
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */

    public void setMatrix(int i0, int i1, int j0, int j1, Matrix X) {
        X.copyTo(0, 0, this, i0, j0, i1 - i0 + 1, j1 - j0 + 1);
    }

    /**
     * Set a submatrix.(There is no optimization for openCl)
     * 
     * @param r
     *            Array of row indices.
     * @param c
     *            Array of column indices.
     * @param X
     *            A(r(:),c(:))
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */

    public void setMatrix(int[] r, int[] c, Matrix X) {
        double[][] A = this.getArrayCopy();
        double[][] XA = X.getArrayCopy();
        try {
            for (int i = 0; i < r.length; i++) {
                for (int j = 0; j < c.length; j++) {
                    A[r[i]][c[j]] = XA[i][j];
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
        this.set(A);
    }

    /**
     * Set a submatrix.(There is no optimization for openCl)
     * 
     * @param r
     *            Array of row indices.
     * @param j0
     *            Initial column index
     * @param j1
     *            Final column index
     * @param X
     *            A(r(:),j0:j1)
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */

    public void setMatrix(int[] r, int j0, int j1, Matrix X) {
        double[][] A = this.getArrayCopy();
        double[][] XA = X.getArrayCopy();
        try {
            for (int i = 0; i < r.length; i++) {
                for (int j = j0; j <= j1; j++) {
                    A[r[i]][j] = XA[i][j - j0];
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
        this.set(A);
    }

    /**
     * Set a submatrix.(There is no optimization for openCl)
     * 
     * @param i0
     *            Initial row index
     * @param i1
     *            Final row index
     * @param c
     *            Array of column indices.
     * @param X
     *            A(i0:i1,c(:))
     * @exception ArrayIndexOutOfBoundsException
     *                Submatrix indices
     */

    public void setMatrix(int i0, int i1, int[] c, Matrix X) {
        double[][] A = this.getArrayCopy();
        double[][] XA = X.getArrayCopy();
        try {
            for (int i = i0; i <= i1; i++) {
                for (int j = 0; j < c.length; j++) {
                    A[i][c[j]] = XA[i - i0][j];
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("Submatrix indices");
        }
        this.set(A);
    }

    /**
     * Read a matrix from a stream. The format is the same the print method, so
     * printed matrices can be read back in (provided they were printed using US
     * Locale). Elements are separated by whitespace, all the elements for each row
     * appear on a single line, the last row is followed by a blank line.
     * 
     * @param input
     *            the input stream.
     */

    public static Matrix read(BufferedReader input) throws java.io.IOException {
        StreamTokenizer tokenizer = new StreamTokenizer(input);

        // Although StreamTokenizer will parse numbers, it doesn't recognize
        // scientific notation (E or D); however, Double.valueOf does.
        // The strategy here is to disable StreamTokenizer's number parsing.
        // We'll only get whitespace delimited words, EOL's and EOF's.
        // These words should all be numbers, for Double.valueOf to parse.

        tokenizer.resetSyntax();
        tokenizer.wordChars(0, 255);
        tokenizer.whitespaceChars(0, ' ');
        tokenizer.eolIsSignificant(true);
        java.util.Vector<Double> vD = new java.util.Vector<Double>();

        // Ignore initial empty lines
        while (tokenizer.nextToken() == StreamTokenizer.TT_EOL)
            ;
        if (tokenizer.ttype == StreamTokenizer.TT_EOF)
            throw new java.io.IOException("Unexpected EOF on matrix read.");
        do {
            vD.addElement(Double.valueOf(tokenizer.sval)); // Read & store 1st row.
        } while (tokenizer.nextToken() == StreamTokenizer.TT_WORD);

        int n = vD.size(); // Now we've got the number of columns!
        double row[] = new double[n];
        for (int j = 0; j < n; j++) // extract the elements of the 1st row.
            row[j] = vD.elementAt(j).doubleValue();
        java.util.Vector<double[]> v = new java.util.Vector<double[]>();
        v.addElement(row); // Start storing rows instead of columns.
        while (tokenizer.nextToken() == StreamTokenizer.TT_WORD) {
            // While non-empty lines
            v.addElement(row = new double[n]);
            int j = 0;
            do {
                if (j >= n)
                    throw new java.io.IOException("Row " + v.size() + " is too long.");
                row[j++] = Double.valueOf(tokenizer.sval).doubleValue();
            } while (tokenizer.nextToken() == StreamTokenizer.TT_WORD);
            if (j < n)
                throw new java.io.IOException("Row " + v.size() + " is too short.");
        }
        int m = v.size(); // Now we've got the number of rows.
        double[][] A = new double[m][];
        v.copyInto(A); // copy the rows out of the vector
        return new Matrix(A);
    }

    /**
     * Print the matrix to the output stream. Line the elements up in columns. Use
     * the format object, and right justify within columns of width characters. Note
     * that is the matrix is to be read back in, you probably will want to use a
     * NumberFormat that is set to US Locale.
     * 
     * @param output
     *            the output stream.
     * @param format
     *            A formatting object to format the matrix elements
     * @param width
     *            Column width.
     * @see java.text.DecimalFormat#setDecimalFormatSymbols
     */
    public void print(PrintWriter output, NumberFormat format, int width) {
        output.println(); // start on new line.
        double data[][] = this.getArrayCopy();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                String s = format.format(data[i][j]); // format the number
                int padding = Math.max(1, width - s.length()); // At _least_ 1 space
                for (int k = 0; k < padding; k++)
                    output.print(' ');
                output.print(s);
            }
            output.println();
        }
        output.println(); // end with blank line.
    }

    /**
     * Print the matrix to stdout. Line the elements up in columns with a
     * Fortran-like 'Fw.d' style format.
     * 
     * @param w
     *            Column width.
     * @param d
     *            Number of digits after the decimal.
     */

    public void print(int w, int d) {
        print(new PrintWriter(System.out, true), w, d);
    }

    /**
     * Print the matrix to stdout. Line the elements up in columns with a
     * Fortran-like 'F7.2' style format.
     */
    public void print() {
        print(7, 2);
    }

    /**
     * Print the matrix to the output stream. Line the elements up in columns with a
     * Fortran-like 'Fw.d' style format.
     * 
     * @param output
     *            Output stream.
     * @param w
     *            Column width.
     * @param d
     *            Number of digits after the decimal.
     */

    public void print(PrintWriter output, int w, int d) {
        DecimalFormat format = new DecimalFormat();
        format.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
        format.setMinimumIntegerDigits(1);
        format.setMaximumFractionDigits(d);
        format.setMinimumFractionDigits(d);
        format.setGroupingUsed(false);
        print(output, format, w + 2);
    }

    /**
     * Print the matrix to stdout. Line the elements up in columns. Use the format
     * object, and right justify within columns of width characters. Note that is
     * the matrix is to be read back in, you probably will want to use a
     * NumberFormat that is set to US Locale.
     * 
     * @param format
     *            A Formatting object for individual elements.
     * @param width
     *            Field width for each column.
     * @see java.text.DecimalFormat#setDecimalFormatSymbols
     */
    public void print(NumberFormat format, int width) {
        print(new PrintWriter(System.out, true), format, width);
    }

    @Override
    public String toString() {
        this.syncFromDevice();
        FloatBuffer buffer = matrixBuffer.getBuffer();
        buffer.position(0);
        String result = "[";
        for (int m = 0; m < M; m++) {
            if (m != 0)
                result += " ";
            result += "[";
            for (int n = 0; n < N; n++) {
                result += String.format("%7.2f", buffer.get());
                if (n != N - 1)
                    result += ", ";
            }
            result += "]";
            if (m != M - 1)
                result += "\n";
            ;
        }
        result += "]\n";
        return result;
    }

    /**
     * 等待队列中的任务全部完成
     */
    public static void finish() {
        queue.finish();
    }

    /**
     * 释放所有OpenCl资源
     */
    public static void releaseAll() {
        gMath.release();
        inited = false;
    }

    /**
     * 释放显存空间
     */
    public void release() {
        // matrixBuffer.getBuffer().clear();
        matrixBuffer.release();
    }

    protected CLBuffer<FloatBuffer> getArg() {
        return matrixBuffer;
    }

    protected static GMath getGMath() {
        return Matrix.gMath;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        this.release();
        Tools.println("一个Matrix对象被回收");
    }

    /**
     * 将数据从主机端同步到设备端
     */
    private void syncToDevice() {
        matrixBuffer.getBuffer().position(0);
        queue.putWriteBuffer(matrixBuffer, true);
    }

    /**
     * 将数据从设备端同步到主机端
     */
    private void syncFromDevice() {
        matrixBuffer.getBuffer().position(0);
        queue.putReadBuffer(matrixBuffer, true);
    }

    /**
     * 创建不合法参数异常，同时释放资源
     * 
     * @param message
     *            包含的信息
     * @return IllegalArgument异常
     */
    private static IllegalArgumentException newIllegalArgumentException(String message) {
        context.release();
        Tools.println("context被成功释放");
        return new IllegalArgumentException(message);
    }
}
