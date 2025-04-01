import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { BrowseService } from 'src/app/core/http/browse/browse.service';

@Component({
    selector: 'app-browse',
    templateUrl: './browse.component.html',
    styleUrls: ['./browse.component.css']
})
export class BrowseComponent implements OnInit {
    bookList: any[] = [];
    backup: any[] = [];
    bookSearch: string = '';
    selectedSort: string = 'popularity';
    displayCount: number = 0;  // controls the number of displayed books

    constructor(private router: Router, private browseService: BrowseService) { }

    ngOnInit() {
        this.fetchBooks(this.selectedSort);
    }

    // Fetch books from the backend with the selected sort criteria.
    fetchBooks(sort: string = 'popularity') {
        this.browseService.listBooks(sort).subscribe({
            next: (response: any) => {
                // Map the backend response to the component's local structure.
                this.bookList = response.map((book: any) => ({
                    id: book.book_id,
                    title: book.title,
                    author: book.authors,
                    image: book.image_url,
                    stars: book.user_rating || 0,
                    wishList: book.in_wishlist
                }));
                // Maintain a backup for search functionality.
                this.backup = [...this.bookList];
                // Set displayCount to one-tenth of the total books.
                this.displayCount = Math.ceil(this.bookList.length * 0.1);
                console.log('Fetched books:', this.bookList);
            },
            error: (err: any) => {
                console.error('Error fetching books:', err);
            }
        });
    }

    // Toggle wishlist status by calling the appropriate backend endpoint.
    addToWishlist(bookId: number) {
        const book = this.bookList.find(b => b.id === bookId);
        if (!book) return;

        if (book.wishList) {
            this.browseService.deleteWishlist(bookId).subscribe({
                next: (response: any) => {
                    book.wishList = false;
                    console.log('Book removed from wishlist:', response);
                },
                error: (err: any) => {
                    console.error('Error removing from wishlist:', err);
                    alert('Error: ' + err.error.message);
                }
            });
        } else {
            this.browseService.addToWishlist(bookId).subscribe({
                next: (response: any) => {
                    book.wishList = true;
                    console.log('Book added to wishlist:', response);
                },
                error: (err: any) => {
                    console.error('Error adding to wishlist:', err);
                    alert('Error: ' + err.error.message);
                }
            });
        }
    }

    // Update the rating for a book using the backend endpoint.
    rateBook(bookId: number, stars: number) {
        this.browseService.editRating(bookId, stars).subscribe({
            next: (response: any) => {
                const book = this.bookList.find(b => b.id === bookId);
                if (book) {
                    book.stars = response.rating;
                }
                console.log('Rating updated:', response);
            },
            error: (err: any) => {
                console.error('Error updating rating:', err);
                alert('Error: ' + err.error.message);
            }
        });
    }

    // Filter the displayed books based on the search input.
    searchBook() {
        this.bookList = this.backup.filter(book =>
            book.title.toLowerCase().includes(this.bookSearch.toLowerCase())
        );
        // Reset displayCount after filtering.
        this.displayCount = Math.ceil(this.bookList.length * 0.1);
    }

    // When the sort criteria change, fetch books sorted by the selected criteria.
    onSortChange(event: any) {
        const value = event.target.value;
        console.log('Sort changed:', value);
        this.selectedSort = value;
        this.fetchBooks(value);
    }

    // Show more books – here we simply reveal all the books.
    showMoreBooks() {
        this.displayCount = this.bookList.length;
    }

    navigateToLibrary() {
        this.router.navigate(['/library']);
    }

    navigateToRecommend() {
        this.router.navigate(['/recommendation']);
    }

    navigateToSettings() {
        this.router.navigate(['/settings']);
    }
}
