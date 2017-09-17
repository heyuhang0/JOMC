package com.hyh0.gmath.samples;

import com.hyh0.gmath.Matrix;
import com.hyh0.gmath.MatrixMath;

class Basic {

    public static void main(String[] args) {
        System.out.println("正在初始化OpenCl....");
        Matrix.init(); // 初始化OpenCl,大约用时5s
        System.out.println("初始化完成\n");

        double[][] data = { { 1, 2, 3 }, { 4, 5, 6 } };
        Matrix A = new Matrix(data); // 通过二维数组创建矩阵
        System.out.println("A = ");
        A.print(); // 输出矩阵

        Matrix B = new Matrix(2, 3, 1); // 创建一个2*3且全部填充1的矩阵
        System.out.println("B = ");
        B.print();
        
        B.set(1, 2, 0.5); // 将 B[1][2] 设置为 0.5
        System.out.println("After edited, B = ");
        B.print();
        
        A.plusEquals(B); // A += B
        System.out.println("After A += B, A = ");
        A.print();

        A.timesEquals(2); // A *= 2
        System.out.println("After A *= 2, A = ");
        A.print();
        
        Matrix D = Matrix.random(3, 3); // 创建一个 3*3 的随机矩阵
        System.out.println("D = ");
        D.print();
        Matrix E = new Matrix(2, 3); // 创建一个 2*3 的矩阵保存结果
        B.times(D, E); // E = B * D
        System.out.println("E = B * D, E = ");
        E.print();

        Matrix F = new Matrix(3, 2);
        E.transpose(F); // F = E'
        System.out.println("E' = ");
        F.print();
        
        F.release(); // 释放F占用的资源
        F = new Matrix(2, 3);
        F = MatrixMath.sigmoid(E, F); // 计算E矩阵的sigmoid值
        System.out.println("F = sigmoid(E), F = ");
        F.print();
        
        Matrix F1 = new Matrix(2, 3);
        F.copyTo(F1); // 将 F 的数据复制到 F1
        System.out.println("F1 = F = ");
        F1.print();
        
        F1 = F.copy(); // 或者直接克隆F
        System.out.println("Second way to let F1 = F = ");
        F1.print();
        
        System.out.println("F is equal to F1, " + F.isEqualTo(F1)); //比较是否相等
        
        System.out.println("A = ");
        A.print();
        System.out.println("B = ");
        B.print();
        System.out.println("A .* B = ");
        A.arrayTimes(B, F).print(); // F = A .* B
        System.out.println("A ./ B = ");
        A.arrayDivides(B, F).print(); // F = A ./ B
        System.out.println("1 ./ A = ");
        A.leftDivide(1, F).print(); // F = 1 ./ A
        
        //矩阵复制操作
        A.release();
        B.release();
        A = Matrix.random(4, 5);
        B = new Matrix(4, 5);
        
        Matrix.releaseAll(); // 释放OpenCl资源
    }

}
