package foo.corporation;

class FooCorporation {

    public static void totalPay(double basepay, double hoursworked){
        if (basepay < 8.0){
            System.out.println("Base pay does not meet minimum wage requirements!");
        }
        else {
            if (hoursworked < 40){
                System.out.println("Employee is owed " + (basepay * hoursworked));
            }
            else if (hoursworked > 40 && hoursworked < 60){
                System.out.println("Employee is owed " + ((basepay * 40) + (basepay * 1.5) * (hoursworked - 40)));
            }
            else {
                System.out.println("Hours worked is greater than 60, errawr!");
            }
        }
    }

    public static void main(String[] args) {
	// write your code here
        double emp1Base = 7.50;
        int emp1hours = 35;
        totalPay(emp1Base, emp1hours);

        double emp2Base = 8.20;
        int emp2hours = 47;
        totalPay(emp2Base, emp2hours);

        double emp3Base = 10.00;
        int emp3hours = 73;
        totalPay(emp3Base, emp3hours);

    }
}
