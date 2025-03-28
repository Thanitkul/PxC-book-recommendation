const { pool } = require("./db");

const getAllBooksForAdmin = async () => {
  try {
    const result = await pool.query("SELECT book_id, title, authors, is_visible FROM app.books");
    if (result.rows.length === 0) {
      return [];
    }
    // Map the result to the desired format
    const books = result.rows.map((book) => ({
      book_id: book.book_id,
      title: book.title,
      authors: book.authors,
      is_visible: book.is_visible,
    }));
    return books;
  } catch (error) {
    console.error("Error fetching all books:", error);
    throw error;
  }
}

// update book visibility and return the updated book
const updateBookVisibility = async (bookId, isVisible) => {
    try {
        const result = await pool.query(
        "UPDATE app.books SET is_visible = $1 WHERE book_id = $2 RETURNING book_id, title, authors, is_visible",
        [isVisible, bookId]
        );
        
        if (result.rows.length === 0) {
            return null; // No book found with the given ID
        }

        let updatedBook = result.rows[0];
        // Convert the updated book to the desired format
        updatedBook = {
            book_id: updatedBook.book_id,
            title: updatedBook.title,
            authors: updatedBook.authors,
            is_visible: updatedBook.is_visible,
        };

        return updatedBook;
    }
    catch (error) {
        console.error("Error updating book visibility:", error);
        throw error;
    }
}

// Export the functions
module.exports = {
    getAllBooksForAdmin,
    updateBookVisibility
};
