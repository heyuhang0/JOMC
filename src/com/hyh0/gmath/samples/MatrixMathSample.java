package com.hyh0.gmath.samples;

import com.hyh0.gmath.Matrix;
import com.hyh0.gmath.MatrixMath;

public class MatrixMathSample {

    public static void main(String[] args) {
        Matrix A = Matrix.random(3, 2, -10, 10);
        Matrix B = new Matrix(3, 2);
        
        A.print();
        B = MatrixMath.abs(A, B);
        B.print();
        
        B = MatrixMath.pow(A, 2, B);
        System.out.println(B);
        B.print();
    }

}
