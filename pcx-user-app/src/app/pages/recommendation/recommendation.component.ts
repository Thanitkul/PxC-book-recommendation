import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from 'src/app/core/http/auth/auth.service';
import { RecommendationService } from 'src/app/core/http/recommendation/recommendation.service';
import { GenresModaService } from 'src/app/modals/genres-modal/genres-moda.service';

@Component({
    selector: 'app-recommendation',
    templateUrl: './recommendation.component.html',
    styleUrls: ['./recommendation.component.css']
})
export class RecommendationComponent {

    // Initialize as empty; will be populated from the backend
    bookList: any[] = [];

    constructor(
        private router: Router,
        private genresService: GenresModaService,
        private authServ: AuthService,
        private recommendationServ: RecommendationService
    ) { }

    ngOnInit() {
        this.getUserInfo();
    }

    getUserInfo() {
        this.authServ.getUserInfo().subscribe({
            next: (response: any) => {
                if (!response.selectedGenre) {
                    this.openGenresModal();
                } else {
                    // Fetch recommendations if the user has already selected genres
                    this.getRecommendations();
                }
            },
            error: (err: any) => {
                console.error('Error fetching user info:', err);
            }
        });
    }

    openGenresModal() {
        this.genresService.openModal().subscribe({
            next: (data: any[]) => {
                this.recommendationServ.selectGenres(data).subscribe({
                    next: () => {
                        // Once the genres are set, fetch the recommendations
                        this.getRecommendations();
                    },
                    error: (err: any) => {
                        alert('Error: ' + err.error.message);
                    }
                });
            }
        });
    }

    // Fetch the list of recommended books from the backend
    getRecommendations() {
        this.recommendationServ.getRecommendations().subscribe({
            next: (response: any) => {
                this.bookList = response.map((book: any) => ({
                    ...book,
                    stars: book.stars || 0, // Default to 0 if no stars are provided
                    wishList: book.wishList || false // Default to false if not in wishlist
                }));
                console.log('Recommended books:', this.bookList);
            },
            error: (err: any) => {
                console.error('Error fetching recommendations:', err);
            }
        });
    }

    // Edit the rating for a book by calling the backend endpoint
    rateBook(bookId: number, stars: number) {
        this.recommendationServ.editRating(bookId, stars).subscribe({
            next: (response: any) => {
                const book = this.bookList.find(b => b.book_id === bookId);
                if (book) {
                    book.rating = response.rating; // Update the book's rating in the local list
                }
                console.log('Rating updated:', response);
            },
            error: (err: any) => {
                console.error('Error updating rating:', err);
            }
        });
    }

    // Add a book to the wishlist via the backend endpoint
    addToWishlist(bookId: number) {
        // Check if the book is already wishlisted
        const book = this.bookList.find(b => b.book_id === bookId);
        if (book && book.wishlisted) {
            this.recommendationServ.removeFromWishlist(bookId).subscribe({
                next: (response: any) => {
                    const book = this.bookList.find(b => b.book_id === bookId);
                    if (book) {
                        book.wishlisted = false;
                    }
                    console.log('Book removed from wishlist:', response);
                },
                error: (err: any) => {
                    console.error('Error removing from wishlist:', err);
                    alert('Error: ' + err.error.message);
                }
            });
        }
        else {
            this.recommendationServ.addToWishlist(bookId).subscribe({
                next: (response: any) => {
                    const book = this.bookList.find(b => b.book_id === bookId);
                    if (book) {
                        book.wishlisted = true;
                    }
                    console.log('Book added to wishlist:', response);
                },
                error: (err: any) => {
                    console.error('Error adding to wishlist:', err);
                    alert('Error: ' + err.error.message);
                }
            });
        }

    }

    navigateToLibrary() {
        this.router.navigate(['/library']);
    }

    navigateToBrowse() {
        this.router.navigate(['/browse']);
    }

    navigateToSettings() {
        this.router.navigate(['/settings']);
    }
}
