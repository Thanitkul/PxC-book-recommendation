import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { LibraryService } from 'src/app/core/http/library/library.service';

@Component({
    selector: 'app-library',
    templateUrl: './library.component.html',
    styleUrls: ['./library.component.css']
})
export class LibraryComponent implements OnInit {

    // Holds the list of rated books (or wishlist books if extended later)
    ratingList: any[] = [];
    wishList: any[] = [];
    ratingBackup: any[] = [];
    wishListBackup: any[] = [];
    page: String = 'rating';
    bookSearch: String = '';


    constructor(
        private router: Router,
        private libraryService: LibraryService
    ) { }

    ngOnInit() {
        this.getListRatings();
        this.getWishList();
    }

    changePage(page: String) {
        this.page = page;
    }

    search() {
        if (this.page == 'rating') {
            this.ratingList = this.ratingBackup.filter(book => book.title.toLowerCase().includes(this.bookSearch.toLowerCase()))
        }
        if (this.page == 'wishlist') {
            this.wishList = this.wishListBackup.filter(book => book.title.toLowerCase().includes(this.bookSearch.toLowerCase()))
        }
    }


    // Fetch the list of rated books from the backend
    getListRatings() {
        this.libraryService.getListRatings().subscribe({
            next: (response: any) => {
                // Map the backend response to your local structure.
                // Expected backend response properties: book_id, rating, created_at, title, authors, image_url
                this.ratingList = response.map((book: any) => ({
                    ...book,
                    rating: book.rating || 0,
                    wishlisted: book.wishlisted || false // if provided by your backend
                }));
                this.ratingBackup = [...this.ratingList]; // Backup the original list for search functionality
            },
            error: (err: any) => {
                console.error('Error fetching rated books:', err);
            }
        });
    }

    getWishList() {
        this.libraryService.getWishlist().subscribe({
            next: (response: any) => {
                // Map the backend response to your local structure.
                // Expected backend response properties: book_id, rating, created_at, title, authors, image_url
                this.wishList = response.map((book: any) => ({
                    ...book,
                    rating: book.rating || 0,
                    wishlisted: book.wishlisted || false // if provided by your backend
                }));
                this.wishListBackup = [...this.wishList]; // Backup the original list for search functionality
            },
            error: (err: any) => {
                console.error('Error fetching rated books:', err);
            }
        });
    }

    // Update the rating for a book using the backend endpoint
    rateBook(bookId: number, stars: number) {
        this.libraryService.editRating(bookId, stars).subscribe({
            next: (response: any) => {
                const book = this.ratingList.find(b => b.book_id === bookId);
                if (book) {
                    book.rating = response.rating; // Update the rating locally with the response
                }
                console.log('Rating updated:', response);
            },
            error: (err: any) => {
                console.error('Error updating rating:', err);
            }
        });
    }

    // Delete a rating for a book using the backend endpoint
    deleteRating(bookId: number) {
        this.libraryService.deleteRating(bookId).subscribe({
            next: (response: any) => {
                // Remove the book from the local list after deletion
                this.ratingList = this.ratingList.filter(b => b.book_id !== bookId);
                this.ratingBackup = this.ratingList; // Update the backup list for search functionality
            },
            error: (err: any) => {
                console.error('Error deleting rating:', err);
            }
        });
    }

    // Remove a book from the wishlist using the backend endpoint
    removeFromWishlist(bookId: number) {
        this.libraryService.deleteWishlist(bookId).subscribe({
            next: (response: any) => {
                // Remove the book from the local wishlist after deletion
                this.wishList = this.wishList.filter(b => b.book_id !== bookId);
                this.wishListBackup = this.wishList; // Update the backup list for search functionality

                console.log('Book removed from wishlist:', response);
            },
            error: (err: any) => {
                console.error('Error removing from wishlist:', err);
            }
        });
    }

    // Navigation methods
    navigateToRecommend() {
        this.router.navigate(['/recommendation']);
    }

    navigateToBrowse() {
        this.router.navigate(['/browse']);
    }

    navigateToSettings() {
        this.router.navigate(['/settings']);
    }
}
