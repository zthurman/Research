package parameter.promiscuosity;

class Square {

    public static void printSquare(int x) {
	// write your code here
        System.out.println(x*x);
    }

    public static void main(String[] args){
        int value = 2;
        printSquare(value);
        printSquare(3);
        printSquare(value*2);
    }
}

class Multiply {

    public static void times(double a, double b){
        System.out.println(a * b);
    }

    public static void main(String[] args){
        times(2,2);
        times(3,4);
    }
}

class Square4 {

    public static double square(double x){
        return x*x;
    }

    public static void main(String[] args){
        System.out.println(square(5));
        System.out.println(square(2));
    }
}

class SquareChange {

    public static void printSquare(int x) {
        System.out.println("printSquare x = " + x);
        x = x * x;
        System.out.println("printSquare x = " + x);
    }

    public static void main(String[] args){
        int x = 5;
        System.out.println("main x = " + x);
        printSquare(x);
        System.out.println("main x = " + x);
    }
}

//class Scope {
//    public static void main(String[] args){
//        int x = 5;
//        if (x ==5){
//            int x = 6;
//            int y = 72;
//            System.out.println("x = " + x + " y = " + y);
//        }
//        System.out.println("x = " + x + " y = " + y);
//    }
//}