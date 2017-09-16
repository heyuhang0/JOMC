package com.hyh0.gmath.samples;

import com.hyh0.gmath.Matrix;
import com.hyh0.gmath.MatrixMath;

class MatrixMathSample {

    public static void main(String[] args) {
        Matrix A = Matrix.random(2, 3, -2, 2);
        Matrix B = new Matrix(2, 3);

        /*
         * 正确性测试
         */
        System.out.println("A = ");
        A.print();

        // 绝对值
        A = MatrixMath.abs(A, A); // A = |A|
        System.out.println("After A = |A|, A = ");
        A.print();

        // 求幂
        B = MatrixMath.pow(A, 2, B); // B = A^2
        System.out.println("A^2 = ");
        B.print();

        B = MatrixMath.pow(Math.E, A, B); // B = e^A
        System.out.println("e^A = exp(A) = ");
        B.print();

        // exp & log 测试
        B = MatrixMath.exp(A, B); // B = exp(A) = e^A
        System.out.println("compute exp(A) directly: exp(A) = ");
        B.print();

        B = MatrixMath.log(B, B); // B = ln(B)
        System.out.println("ln(exp(A)) = ");
        B.print();
        
        // exp2 & log2 测试
        B = MatrixMath.exp2(A, B); // B = exp(A) = e^A
        System.out.println("exp2(A) = ");
        B.print();

        B = MatrixMath.log2(B, B); // B = ln(B)
        System.out.println("log2(exp2(A)) = ");
        B.print();
        
        // exp10 & log10 测试
        B = MatrixMath.exp10(A, B); // B = exp(A) = e^A
        System.out.println("exp10(A) = ");
        B.print();

        B = MatrixMath.log10(B, B); // B = ln(B)
        System.out.println("log10(exp10(A)) = ");
        B.print();
        
        // sqrt & pow 测试
        B = MatrixMath.pow(A, 0.5, B); // B = A^0,5
        System.out.println("A^0.5 = ");
        B.print();
        
        B = MatrixMath.sqrt(A, B); // B = sqrt(A)
        System.out.println("sqrt(A) = ");
        B.print();
        
        B = MatrixMath.rsqrt(A, B); // B = 1/sqrt(A)
        System.out.println("1/sqrt(A) = ");
        B.print();
        
        // sin & asin 测试
        A.release();
        A = Matrix.random(2, 3);
        System.out.println("now A = ");
        A.print();
        
        B = MatrixMath.sinh(A, B);
        System.out.println("sinh(A) = ");
        B.print();
        
        B = MatrixMath.asin(A, B);
        System.out.println("asin(A) = ");
        B.print();
        
        B = MatrixMath.sin(B, B);
        System.out.println("sin(asin(A)) = ");
        B.print();
        
        // cos & acos 测试
        B = MatrixMath.cosh(A, B);
        System.out.println("cosh(A) = ");
        B.print();
        
        B = MatrixMath.acos(A, B);
        System.out.println("acos(A) = ");
        B.print();
        
        B = MatrixMath.cos(B, B);
        System.out.println("cos(acos(A)) = ");
        B.print();
        
        // tan & atan 测试
        B = MatrixMath.tanh(A, B);
        System.out.println("tanh(A) = ");
        B.print();
        
        B = MatrixMath.atan(A, B);
        System.out.println("atan(A) = ");
        B.print();
        
        B = MatrixMath.tan(B, B);
        System.out.println("tan(atan(A)) = ");
        B.print();
    }

}
