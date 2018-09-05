package loop.array;

class LoopArray {

    public static void main(String[] args){
        int[] values = new int[5];

        for (int i=0; i<values.length; i++){
            values[i] = i;
            int y = values[i] * values[i];
            System.out.println(y);
        }
    }

}

class LoopArray2{

    public static void main(String[] args){
        int[] values = new int[5];
        int i = 0;
        while (i < values.length){
            values[i] = i;
            int y = values[i] * values[i];
            System.out.println(y);
            i++;
        }
    }
}
