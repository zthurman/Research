package marathon.sorter;

class Marathon {

    public static int getMinIndex(int[] values){
        int minValue = Integer.MAX_VALUE;
        int minIndex = -1;

        for (int i=0; i < values.length; i++){
            if (values[i] < minValue) {
                minValue = values[i];
                minIndex = i;
            }
        }
        return minIndex;
    }

    public static int getSecondMinIndex(int[] values){
        int secondIdx = -1;
        int minIdx = getMinIndex(values);

        for (int i=0; i< values.length; i++){
            if (i == minIdx){
                continue;
            }
            if (secondIdx == -1 || values[i] < values[secondIdx]) {
                secondIdx = i;
            }
        }
        return secondIdx;
    }

    public static void printFastest(String[] names, int[] times){
        int min = getMinIndex(times);
        int secondmin = getSecondMinIndex(times);
        System.out.println(names[min] + " is the fastest with a time of " + times[min]);
        System.out.println(names[secondmin] + " is the second fastest with a time of " + times[secondmin]);
    }

    public static void main(String[] args) {
	// write your code here
        String[] names = {
                "Elena", "Thomas", "Hamilton", "Suzie", "Phil", "Matt", "Alex",
                "Emma", "John", "James", "Jane", "Emily", "Daniel", "Neda",
                "Aaron", "Kate"
        };

        int[] times = {
                341, 273, 278, 329, 445, 402, 388, 275, 243, 334, 412, 393, 299,
                343, 317, 265
        };

        printFastest(names, times);

    }
}
