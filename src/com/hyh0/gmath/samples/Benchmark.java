package com.hyh0.gmath.samples;

import com.hyh0.gmath.Matrix;
import com.hyh0.gmath.MatrixMath;

class Benchmark {

    public static void main(String[] args) {
        double time;
        
        System.out.println("正在初始化OpenCl....");
        Matrix.init(); // 初始化OpenCl,大约用时5s
        System.out.println("初始化完成\n");
        
        resetTimer();
        Matrix miniA = Matrix.random(100, 100);
        Matrix miniB = Matrix.random(100, 100);
        Matrix miniC = Matrix.random(100, 100);
        Matrix.finish();
        System.out.println("产生3个100*100的随机矩阵用时: ");
        showTimer();
        System.out.println();
        
        resetTimer();
        Matrix A = Matrix.random(3000, 3000); // 产生 3000 * 3000 的随机矩阵
        Matrix B = Matrix.random(3000, 3000);
        Matrix C = Matrix.random(3000, 3000);
        Matrix.finish(); // 等待队列中的任务完成
        System.out.println("产生3个3000*3000的随机矩阵用时: ");
        showTimer();
        System.out.println();
        
        resetTimer();
        A.times(B, C);
        Matrix.finish();
        System.out.println("3000*3000 的矩阵相乘用时: ");
        time = showTimer();
        System.out.println("浮点运算速度为: ");
        System.out.println((double)3000*3000*3000*2*1000/1e9/time + " GFLOPS");
        System.out.println();
        
        resetTimer();
        for(int i = 0; i < (int)1e4; i++) {
            miniA.times(miniB, miniC);
        }
        Matrix.finish();
        System.out.println("10000次100*100的矩阵相乘用时: ");
        time = showTimer();
        System.out.println("浮点运算速度为: ");
        System.out.println((double)100*100*100*2*1e4*1000/1e9/time + " GFLOPS");
        System.out.println();
        
        resetTimer();
        A.plus(B, C);
        Matrix.finish();
        System.out.println("3000*3000 的矩阵相加用时: ");
        time = showTimer();
        System.out.println("浮点运算速度为: ");
        System.out.println((double)3000*3000*1000/1e9/time + " GFLOPS");
        System.out.println();
        
        resetTimer();
        for(int i = 0; i < (int)1e4; i++) {
            miniA.plus(miniB, miniC);
        }
        Matrix.finish();
        System.out.println("10000次100*100的矩阵相加用时: ");
        time = showTimer();
        System.out.println("浮点运算速度为: ");
        System.out.println((double)100*100*1e4*1000/1e9/time + " GFLOPS");
        System.out.println();
        
        resetTimer();
        A.times(2.45, C);
        Matrix.finish();
        System.out.println("3000*3000 的矩阵数乘用时: ");
        time = showTimer();
        System.out.println("浮点运算速度为: ");
        System.out.println((double)3000*3000*1000/1e9/time + " GFLOPS");
        System.out.println();
        
        resetTimer();
        for(int i = 0; i < (int)1e4; i++) {
            miniA.times(2.4212, miniB);
        }
        Matrix.finish();
        System.out.println("10000次100*100的矩阵数乘用时: ");
        time = showTimer();
        System.out.println("浮点运算速度为: ");
        System.out.println((double)100*100*1e4*1000/1e9/time + " GFLOPS");
        System.out.println();
        
        resetTimer();
        C = MatrixMath.sigmoid(A, C);
        Matrix.finish();
        System.out.println("3000*3000 的矩阵sigmoid用时: ");
        showTimer();
        System.out.println();
        
        Matrix.releaseAll();
    }

    static long timer;
    /**
     * 重置计时器
     */
    public static void resetTimer() {
        timer = System.nanoTime();
    }
    /**
     * 显示当前与上次重置计时器之间的时间差(Ms)
     * @return 时间差(Ms)
     */
    public static double showTimer() {
        double time = (double)(System.nanoTime() - timer)/1000000;
        System.out.println(time + "ms");
        resetTimer();
        return time;
    }
}
