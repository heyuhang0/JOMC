package com.hyh0.gmath;

import com.hyh0.gmath.debug.Tools;

public class GMathTest {
    
    public static void main(String[] args) {
        GMath gpu = new GMath();
        speedTest(1000, 1000, 1000, gpu);
        
        speedTest(10, 2000, 2000, gpu);
        speedTest(2000, 10, 2000, gpu);
        speedTest(2000, 2000, 10, gpu);
        
        speedTest(10, 10, 50000, gpu);
        speedTest(50000, 10, 10, gpu);
        speedTest(10, 50000, 10, gpu);
        
        GMatrix mA = gpu.newGMatrix(10, 10);
        GMatrix mB = gpu.newGMatrix(10, 10);
        GMatrix mC1 = gpu.newGMatrix(10, 10);
        GMatrix mC2 = gpu.newGMatrix(10, 10);
        gpu.fillMatrixRandomly(mA, -5, 5);
        gpu.fillMatrixRandomly(mB, -5, 5);
        gpu.multiply(mA, mB, mC1);
        gpu.multiply2(mA, mB, mC2);
        System.out.println(mC1);
        System.out.println(mC2);
        
        gpu.release();
    }
    
    static void speedTest(int m, int n, int p, GMath gpu) {
        GMatrix mA = gpu.newGMatrix(m, n);
        GMatrix mB = gpu.newGMatrix(n, p);
        GMatrix mC = gpu.newGMatrix(m, p);
        
        gpu.fillMatrixRandomly(mA, -5, 5);
        gpu.fillMatrixRandomly(mB, -5, 5);
        
        Tools.resetTimer();
        gpu.multiply(mA, mB, mC);
        gpu.finish();
        System.out.print("m=" + m + " n=" + n + " p=" + p + " : ");
        Tools.showTimer();
        
        Tools.resetTimer();
        gpu.multiply2(mA, mB, mC);
        gpu.finish();
        System.out.print("m=" + m + " n=" + n + " p=" + p + " : ");
        Tools.showTimer();
        
        mA.release();
        mB.release();
        mC.release();
    } 
}
