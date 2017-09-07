package com.hyh0.gmath.debug;

public class Tools {
    
    final static boolean showMessages = true;
    public static void println(String message) {
        if(showMessages)
            System.out.println("$: " + message);
    }
    public static void println(Object o) {
        Tools.println(o.toString());
    }
    
    static long timer;
    public static void resetTimer() {
        timer = System.nanoTime();
    }
    public static void showTimer() {
        System.out.println((double)(System.nanoTime() - timer)/1000000 + "ms");
        Tools.resetTimer();
    }
}
