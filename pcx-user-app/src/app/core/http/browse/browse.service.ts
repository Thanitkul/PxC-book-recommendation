import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment.development';

@Injectable({
    providedIn: 'root'
})
export class BrowseService {
    private baseUrl = `${environment.apiHost}/api/browse`;

    constructor(private http: HttpClient) { }

    // GET /api/browse/list-books?sort=popularity
    listBooks(sort: string = 'popularity') {
        return this.http.get(`${this.baseUrl}/list-books?sort=${sort}`);
    }

    // PATCH /api/browse/edit-rating
    editRating(book_id: number, rating: number) {
        return this.http.patch(`${this.baseUrl}/edit-rating`, { book_id, rating });
    }

    // DELETE /api/browse/delete-rating
    deleteRating(book_id: number) {
        return this.http.delete(`${this.baseUrl}/delete-rating`, { body: { book_id } });
    }

    // POST /api/browse/add-wishlist
    addToWishlist(book_id: number) {
        return this.http.post(`${this.baseUrl}/add-wishlist`, { book_id });
    }

    // DELETE /api/browse/delete-wishlist
    deleteWishlist(book_id: number) {
        return this.http.delete(`${this.baseUrl}/delete-wishlist`, { body: { book_id } });
    }
}
