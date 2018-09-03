package com.company;

class HelloWorld {

    public static void main(String[] args) {
	// write your code here
        System.out.println("Hello World!");
    }
}

class Hello2 {

    public static void main(String[] args) {
        // write your code here
        System.out.println("Hello World!");
        System.out.println("Line Number 2!");
    }
}

class Hello3 {

    public static void main(String[] args) {
        // write your code here
        String foo = "IAP 6.092";
        System.out.println(foo);
        foo = "Something else";
        System.out.println(foo);
    }
}

class DoMath {

    public static void main(String[] args) {
        double score = 1.0 + 2.0 * 3.0;
        System.out.println(score);
        score = score / 2.0;
        System.out.println(score);
    }
}

class DoMath2 {

    public static void main(String[] args){
        double score = 1.0 + 2.0 * 3.0;
        System.out.println(score);
        double copy = score;
        copy = copy / 2.0;
        System.out.println(copy);
        System.out.println(score);
    }
}

class StringConcat {

    public static void main(String[] args){
        String text = "hello" + " world";
        text = text + " number " + 5;
        System.out.println(text);
    }
}