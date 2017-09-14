
# OpenCL_Matrix_Computing
用Java实现的基于OpenCl的矩阵运算库，使用了类似JAMA的接口。

## 使用示例

Matrix.init(); // 初始化OpenCl,大约用时5s

double[]
[] data = { { 1, 2, 3 }, { 4, 5, 6 } };

Matrix A = new Matrix(data); // 通过二维数组创建矩阵

Matrix B = new Matrix(2, 3, 1); // 创建一个2*3且全部填充1的矩阵

B.set(1, 2, 0.5); // 将 B(1)(2) 设置为 0.5

A.plusEquals(B); // A += B

A.print(); // 输出矩阵

Matrix.releaseAll();// 释放资源


## 性能
测试平台: Core M 5Y10c @0.8GHz & HD5300

测试样本: 3000*3000矩阵乘法

JAMA: 45s

Octave: 1.8s

本库(CPU): 4.8s

本库(GPU): 3.6s

测试样本: 3000*3000矩阵加法

JAMA: 0.08s

Octave: 0.05s

本库(CPU): 0.012s

本库(GPU): 0.012s

## 当前实现的功能
### 对象操作
constructor

set elements

get elements

copy

clone

print

 
### 矩阵运算
addition

subtraction

multiplication

scalar multiplication 

transpose

sigmoid

 
### OpenCl管理
initialize

release

finish

sync
 
	
## 使用方法
项目使用eclipse开发，UTF-8编码

如果要使用本库，可以用eclipse导入，然后在需要的项目中添加 required project
