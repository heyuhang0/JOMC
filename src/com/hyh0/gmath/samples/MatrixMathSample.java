package com.hyh0.gmath.samples;

import com.hyh0.gmath.Matrix;
import com.hyh0.gmath.MatrixMath;

class MatrixMathSample {

    public static void main(String[] args) {
        Matrix A = Matrix.random(2, 3, -2, 2);
        Matrix B = new Matrix(2, 3);

        System.out.println("A = ");
        A.print();

        A = MatrixMath.abs(A, A); // A = |A|
        System.out.println("After A = |A|, A = ");
        A.print();

        B = MatrixMath.pow(A, 2, B); // B = A^2
        System.out.println("A^2 = ");
        B.print();
        
        B = MatrixMath.exp(A, B); // B = exp(A) = e^A
        System.out.println("exp(A) = ");
        B.print();
        
        B = MatrixMath.pow(Math.E, A, B); // B = e^A
        System.out.println("Another way to compute exp(A) = ");
        B.print();
        
        B = MatrixMath.log(A, B); // B = ln(A)
        System.out.println("ln(A) = ");
        B.print();
    }

}
