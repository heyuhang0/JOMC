package com.hyh0.gmath;

import com.hyh0.gmath.debug.Tools;

public class GMathTest {
    
    public static void main(String[] args) {
        GMath gpu = new GMath();
        
        GMatrix mA = gpu.newGMatrix(1000, 1000);
        GMatrix mB = gpu.newGMatrix(1000, 1000);
        GMatrix mC = gpu.newGMatrix(1000, 1000);
        Tools.resetTimer();
        gpu.fillMatrixRandomly(mA, -1, 1);
        gpu.fillMatrixRandomly(mB, -1, 1);
        
        Tools.showTimer();
        gpu.matrixMultiply(mA, mB, mC);
        gpu.finish();
        Tools.showTimer();
        
        double[] mta = new double[1000000];
        double[] mtb = new double[1000000];
        double[] mtc = new double[1000000];
        Tools.resetTimer();
        for(int i = 0; i < 1000000; i++) {
            mta[i] = Math.random();
            mtb[i] = Math.random();
        }
        Tools.showTimer();
        for(int m = 0; m < 1000; m++)
            for(int n = 0; n < 1000; n++) {
                double sum = 0;
                for(int i = 0; i < 1000; i++) {
                    sum += mta[m * 1000 + i] * mtb[n + i * 1000];
                }
                mtc[m * 1000 + n] = sum;
            }
        Tools.showTimer();
        //System.out.println(mC);
        
        gpu.release();
    }
}
