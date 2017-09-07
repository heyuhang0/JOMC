package com.hyh0.gmath;

import com.hyh0.gmath.debug.Tools;

public class GMathTest {
    
    public static void main(String[] args) {
        GMath gpu = new GMath();
        
        GMatrix mA = gpu.newGMatrix(10000, 1000);
        GMatrix mB = gpu.newGMatrix(10000, 1000);
        gpu.fillMatrixRandomly(mA, -5, 5);
        
        Tools.resetTimer();
        gpu.sigmoid(mA, mB);
        gpu.finish();
        Tools.showTimer();
        double a = 100;
        for(int i = 0; i < 10000000; i++) {
            a = sigmoid((double)i/1000000);
        }
        Tools.showTimer();
        gpu.release();
    }
    
    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
