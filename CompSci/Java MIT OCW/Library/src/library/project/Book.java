package library.project;

public class Book {
    String title;
    boolean borrowed;

    public Book(String bookTitle){
        title = bookTitle;
    }

    public void borrowed() {
        borrowed = true;
    }

    public void returned() {
        borrowed = false;
    }

    public boolean isBorrowed() {
        if (borrowed == true){
            return true;
        } else {
            return false;
        }
    }

    public String getTitle() {
        return title;
    }

    public static void main (String[] args) {
        Book example = new Book("The Da Vinci Code");
        System.out.println("Title (should be The Da Vinci Code): " + example.getTitle());
        System.out.println("Borrowed? (should be false): " + example.isBorrowed());
        example.borrowed();
        System.out.println("Borrowed? (should be true): " + example.isBorrowed());
        example.returned();
        System.out.println("Borrowed? (should be false): " + example.isBorrowed());
    }
}
