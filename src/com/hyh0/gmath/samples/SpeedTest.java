package com.hyh0.gmath.samples;

import com.hyh0.gmath.Matrix;

class SpeedTest {

    public static void main(String[] args) {
        System.out.println("正在初始化OpenCl....");
        Matrix.init(); // 初始化OpenCl,大约用时5s
        System.out.println("初始化完成\n");
        
        resetTimer();
        Matrix A = Matrix.random(3000, 3000); // 产生 3000 * 3000 的随机矩阵
        Matrix B = Matrix.random(3000, 3000);
        Matrix C = Matrix.random(3000, 3000);
        Matrix.finish(); // 等待队列中的任务完成
        System.out.println("产生3个3000*3000的随机矩阵用时: ");
        showTimer();
        System.out.println();
        
        A.times(B, C);
        Matrix.finish();
        System.out.println("A * B 用时: ");
        showTimer();
        System.out.println();
        
        A.plus(B, C);
        Matrix.finish();
        System.out.println("A + B 用时: ");
        showTimer();
        System.out.println();
        
        A.times(2, C);
        Matrix.finish();
        System.out.println("A * 2 用时: ");
        showTimer();
        System.out.println();
        
        A.sigmoid(C);
        Matrix.finish();
        System.out.println("sigmoid(A) 用时: ");
        showTimer();
        System.out.println();
        
        Matrix.releaseAll();
    }

    static long timer;
    public static void resetTimer() {
        timer = System.nanoTime();
    }
    public static void showTimer() {
        System.out.println((double)(System.nanoTime() - timer)/1000000 + "ms");
        resetTimer();
    }
}
