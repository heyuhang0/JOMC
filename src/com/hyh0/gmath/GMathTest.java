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
        
        gpu.release();
    }
    
    static void speedTest(int m, int n, int p, GMath gpu) {
        GMatrix mA = gpu.newGMatrix(m, n);
        GMatrix mB = gpu.newGMatrix(n, p);
        GMatrix mC1 = gpu.newGMatrix(m, p);
        GMatrix mC2 = gpu.newGMatrix(m, p);
        
        gpu.fillMatrixRandomly(mA, -5, 5);
        gpu.fillMatrixRandomly(mB, -5, 5);
        
        Tools.resetTimer();
        gpu.multiply(mA, mB, mC1);
        gpu.finish();
        System.out.print("m=" + m + " n=" + n + " p=" + p + " : ");
        Tools.showTimer();
        
        Tools.resetTimer();
        gpu.multiply2(mA, mB, mC2);
        gpu.finish();
        System.out.print("m=" + m + " n=" + n + " p=" + p + " : ");
        Tools.showTimer();
        
        System.out.println(gpu.isEqual(mC1, mC2));
        mA.release();
        mB.release();
        mC1.release();
        mC2.release();
    }
    
}
