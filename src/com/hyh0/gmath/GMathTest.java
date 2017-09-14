package com.hyh0.gmath;

import static com.hyh0.gmath.debug.Tools.*;

public class GMathTest {
    
    public static void main(String[] args) {
        GMath gpu = new GMath();

        Matrix mA = gpu.newMatrix(1000);
        Matrix mB = gpu.newMatrix(1000);
        Matrix mC = gpu.newMatrix(1000);
        
        mA.randomize();
        mB.randomize();
        mC.randomize();
        gpu.finish();
        
        gpu.release();
    }
}
