import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment.development';

@Injectable({
    providedIn: 'root'
})
export class LibraryService {
    private baseUrl = `${environment.apiHost}/api/library`;

    constructor(private http: HttpClient) { }

    // GET /api/library/list-ratings
    getListRatings() {
        return this.http.get(`${this.baseUrl}/list-ratings`);
    }

    // PATCH /api/library/edit-ratings
    editRating(book_id: number, rating: number) {
        return this.http.patch(`${this.baseUrl}/edit-ratings`, { book_id, rating });
    }

    // DELETE /api/library/delete-ratings
    deleteRating(book_id: number) {
        return this.http.delete(`${this.baseUrl}/delete-ratings`, { body: { book_id } });
    }

    // GET /api/library/list-wishlist
    getWishlist() {
        return this.http.get(`${this.baseUrl}/list-wishlist`);
    }

    // DELETE /api/library/delete-wishlist
    deleteWishlist(book_id: number) {
        return this.http.delete(`${this.baseUrl}/delete-wishlist`, { body: { book_id } });
    }
}
