package library.project;

import java.util.ArrayList;

public class Library {

    String address ;
    static String openinghours = "Libraries are open daily from 9AM to 5PM.";
    ArrayList<Book> books;

    public Library(String libraryAddress){
        address = libraryAddress;
        books = new ArrayList<Book>();
    }

    public void printAddress() {
        System.out.println(address);
    }

    public static void printOpeningHours(){
        System.out.println(openinghours);
    }

    public void addBook(Book book){

        books.add(book);
    }

    public String borrowBook(String bookTitle) {
        Book libraryBook;
        String libraryBookTitle;

        for(int i = 0; i < books.size(); i+=1)
        {
            libraryBook = books.get(i);
            libraryBookTitle = libraryBook.getTitle();

            if(libraryBookTitle.equals(bookTitle))
            {
                if(!(libraryBook.isBorrowed()))
                {
                    libraryBook.borrowed();
                    System.out.println("You successfully borrowed " + libraryBookTitle);
                    return null;
                }
                else
                {
                    System.out.println("Sorry, this book is already borrowed.");
                    return null;
                }
            }
        }

        System.out.println("Sorry, this book is not in our catalog.");
        return null;
    }

    public void printAvailableBooks()
    {
        Book libraryBook;
        boolean libraryIsEmpty = true;

        for(int i = 0; i < books.size(); i+=1)
        {
            libraryBook = books.get(i);

            if(!(libraryBook.isBorrowed()))
            {
                System.out.println(libraryBook.getTitle());
                libraryIsEmpty = false;
            }
        }

        if(libraryIsEmpty)
        {
            System.out.println("No books in catalog.");
        }
    }

    public void returnBook(String bookTitle){
        Book libraryBook;
        String libraryBookTitle;
        Boolean found = false;

        for (int i=0; i < books.size(); i++){
            libraryBook = books.get(i);
            libraryBookTitle = libraryBook.getTitle();

            if(libraryBookTitle.equals(bookTitle)){
                if(libraryBook.isBorrowed()){
                    libraryBook.returned();
                    System.out.println("You successfully returned: " + libraryBookTitle);
                    found = true;
                    break;
                }
            }
        }

        if(!found){
            System.out.println("Your book was not found in the library catalog.");
        }
    }

    public static void main(String[] args) {
	// write your code here
        Library firstLibrary = new Library("10 Main St.");
        Library secondLibrary = new Library("228 Liberty St.");

        firstLibrary.addBook(new Book("The Da Vinci Code"));
        firstLibrary.addBook(new Book("Le Petit Prince"));
        firstLibrary.addBook(new Book("A Tale of Two Cities"));
        firstLibrary.addBook(new Book("The Lord of the Rings"));

        System.out.println("Library hours: ");
        printOpeningHours();
        System.out.println();

        System.out.println("Library addresses: ");
        firstLibrary.printAddress();
        secondLibrary.printAddress();
        System.out.println();

        System.out.println("Borrowing The Lord of the Rings:");
        firstLibrary.borrowBook("The Lord of the Rings");
        firstLibrary.borrowBook("The Lord of the Rings");
        secondLibrary.borrowBook("The Lord of the Rings");
        System.out.println();

        System.out.println("Books available in the first library:");
        firstLibrary.printAvailableBooks();
        System.out.println();
        System.out.println("Books available in the second library:");
        secondLibrary.printAvailableBooks();

        System.out.println("Returning The Lord of the Rings:");
        firstLibrary.returnBook("The Lord of the Rings");
        System.out.println();

        System.out.println("Books available in the first library:");
        firstLibrary.printAvailableBooks();

    }
}
