import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment.development';

@Injectable({
  providedIn: 'root'
})
export class RecommendationService {
    private baseUrl = `${environment.apiHost}/api/recommend`;

    constructor(private http: HttpClient) { }

    // GET /api/recommend/genre-choices
    getGenreChoices() {
        return this.http.get(`${this.baseUrl}/genre-choices`);
    }

    // POST /api/recommend/select-genres
    selectGenres(selected_genres: any[]) {
        return this.http.post(`${this.baseUrl}/select-genres`, { selected_genres });
    }

    // GET /api/recommend/list
    getRecommendations() {
        return this.http.get(`${this.baseUrl}/list`);
    }

    // PATCH /api/recommend/edit-rating
    editRating(book_id: number, rating: number) {
        return this.http.patch(`${this.baseUrl}/edit-rating`, { book_id, rating });
    }

    // DELETE /api/recommend/delete-rating
    deleteRating(book_id: number) {
        return this.http.delete(`${this.baseUrl}/delete-rating`, { body: { book_id } });
    }

    // POST /api/recommend/add-wishlist
    addToWishlist(book_id: number) {
        return this.http.post(`${this.baseUrl}/add-wishlist`, { book_id });
    }

    // DELETE /api/recommend/delete-wishlist
    removeFromWishlist(book_id: number) {
        return this.http.delete(`${this.baseUrl}/delete-wishlist`, { body: { book_id } });
    }
}
