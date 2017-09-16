package com.hyh0.gmath;

public class MatrixMath {
    private static GMath gMath = null;

    /**
     * 初始化矩阵数学库 (只应该由Matrix库调用)
     * 
     * @param gMath
     */
    protected static void init(GMath gMath) {
        MatrixMath.gMath = gMath;
    }

    /**
     * 计算当前矩阵的sigmoid值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix sigmoid(Matrix input, Matrix result) {
        gMath.sigmoid(input, result);
        return result;
    }

    /**
     * 计算当前矩阵的绝对值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix abs(Matrix input, Matrix result) {
        gMath.abs(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的反余弦值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix acos(Matrix input, Matrix result) {
        gMath.acos(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的反正弦值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix asin(Matrix input, Matrix result) {
        gMath.asin(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的反正切值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix atan(Matrix input, Matrix result) {
        gMath.atan(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的sin值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix cos(Matrix input, Matrix result) {
        gMath.cos(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的cos值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix sin(Matrix input, Matrix result) {
        gMath.sin(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的tan值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix tan(Matrix input, Matrix result) {
        gMath.tan(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的双曲余弦值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix cosh(Matrix input, Matrix result) {
        gMath.cosh(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的双曲正弦值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix sinh(Matrix input, Matrix result) {
        gMath.sinh(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的双曲正切值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix tanh(Matrix input, Matrix result) {
        gMath.tanh(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的以e为底的对数值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix log(Matrix input, Matrix result) {
        gMath.log(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的以2为底的对数值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix log2(Matrix input, Matrix result) {
        gMath.log2(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的以10为底的对数值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix log10(Matrix input, Matrix result) {
        gMath.log10(input, result);
        return result;
    }

    /**
     * 求e的幂
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix exp(Matrix input, Matrix result) {
        gMath.exp(input, result);
        return result;
    }

    /**
     * 求2的幂
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix exp2(Matrix input, Matrix result) {
        gMath.exp2(input, result);
        return result;
    }

    /**
     * 求10的幂
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix exp10(Matrix input, Matrix result) {
        gMath.exp10(input, result);
        return result;
    }

    /**
     * 求每个元素的平方根
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix sqrt(Matrix input, Matrix result) {
        gMath.sqrt(input, result);
        return result;
    }

    /**
     * 求每个元素平方根的导数(1/sqrt(x))
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix rsqrt(Matrix input, Matrix result) {
        gMath.rsqrt(input, result);
        return result;
    }

    /**
     * 求以矩阵元素为底, power为指数的幂
     * 
     * @param input
     *            输入矩阵(充当底数)
     * @param power
     *            指数
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix pow(Matrix input, double power, Matrix result) {
        gMath.pow(input, power, result);
        return result;
    }

    /**
     * 求以base为底, 矩阵元素为指数的幂
     * 
     * @param base
     *            幂运算的底数
     * @param input
     *            输入矩阵(充当指数)
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix pow(double base, Matrix input, Matrix result) {
        gMath.pow(base, input, result);
        return result;
    }

    /**
     * 求幂
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     */
    public static Matrix pow(Matrix input, int power, Matrix result) {
        gMath.pow(input, power, result);
        return result;
    }

    /**
     * 计算当前矩阵的sigmoid值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix sigmoid(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.sigmoid(input, result);
        return result;
    }

    /**
     * 计算当前矩阵的绝对值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix abs(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.abs(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的反余弦值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix acos(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.acos(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的反正弦值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix asin(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.asin(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的反正切值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix atan(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.atan(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的sin值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix cos(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.cos(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的cos值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix sin(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.sin(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的tan值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix tan(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.tan(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的双曲余弦值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix cosh(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.cosh(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的双曲正弦值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix sinh(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.sinh(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的双曲正切值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix tanh(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.tanh(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的以e为底的对数值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix log(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.log(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的以2为底的对数值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix log2(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.log2(input, result);
        return result;
    }

    /**
     * 计算矩阵中每个元素的以10为底的对数值
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix log10(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.log10(input, result);
        return result;
    }

    /**
     * 求e的幂
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix exp(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.exp(input, result);
        return result;
    }

    /**
     * 求2的幂
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix exp2(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.exp2(input, result);
        return result;
    }

    /**
     * 求10的幂
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix exp10(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.exp10(input, result);
        return result;
    }

    /**
     * 求每个元素的平方根
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix sqrt(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.sqrt(input, result);
        return result;
    }

    /**
     * 求每个元素平方根的导数(1/sqrt(x))
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix rsqrt(Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.rsqrt(input, result);
        return result;
    }

    /**
     * 求幂
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix pow(Matrix input, double power) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.pow(input, power, result);
        return result;
    }

    /**
     * 求幂
     * 
     * @param input
     *            输入矩阵
     * @param result
     *            保存运算结果的矩阵
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix pow(Matrix input, int power) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.pow(input, power, result);
        return result;
    }

    /**
     * 求以base为底, 矩阵元素为指数的幂
     * 
     * @param base
     *            幂运算的底数
     * @param input
     *            输入矩阵(充当指数)
     * @return 保存运算结果的矩阵
     * @deprecated 这样使用可能会导致显存泄露，应该把保存结果的矩阵传入方法
     */
    public static Matrix pow(double base, Matrix input) {
        Matrix result = new Matrix(input.getRowDimension(), input.getColumnDimension());
        gMath.pow(base, input, result);
        return result;
    }
}
