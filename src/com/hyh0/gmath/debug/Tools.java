package com.hyh0.gmath.debug;

import java.io.IOException;

public class Tools {
    
    
    final static boolean showMessages = false;
    public static void println(String message) {
        if(showMessages)
            System.out.println("$: " + message);
    }
    public static void println(Object o) {
        Tools.println(o.toString());
    }
    
    static long[] timer = new long[10];
    public static void resetTimer(int index) {
        timer[index] = System.nanoTime();
    }
    public static void resetTimer() {
        resetTimer(9);
    }
    public static void showTimer(int index) {
        System.out.println((double)(System.nanoTime() - timer[index])/1000000 + "ms");
        Tools.resetTimer(index);
    }
    public static void showTimer() {
        showTimer(9);
    }
    
    public static void interrupt(String message) {
        if(message != null)
            System.out.println(message);
        System.out.println("按下回车继续运行...");
        try {
            System.in.read();
            System.in.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static void interrupt() {
        interrupt(null);
    }
}
