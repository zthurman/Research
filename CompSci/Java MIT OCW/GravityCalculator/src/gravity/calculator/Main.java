package gravity.calculator;

class GravityCalculator {

    public static void main(String[] args) {
	// write your code here
        double gravity = -9.81;
        double initialVelocity = 0.0;
        double fallingTime = 10.0;
        double initialPosition = 0.0;
        double finalPosition = 0.0;
        System.out.println("The object's position after " + fallingTime + " seconds is "
        + finalPosition + " m. ");
    }
}

class GravityCalculator2 {

    public static void main(String[] args) {
        // write your code here
        double gravity = -9.81;
        double initialVelocity = 0.0;
        double fallingTime = 10.0;
        double initialPosition = 0.0;
        double finalPosition = 0.0;
        finalPosition = 0.5 * gravity*Math.pow(fallingTime, 2.0) + initialVelocity*fallingTime + initialPosition;
        System.out.println("The object's position after " + fallingTime + " seconds is "
                + finalPosition + " m. ");
    }
}