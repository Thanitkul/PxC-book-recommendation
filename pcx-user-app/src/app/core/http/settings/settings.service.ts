import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment.development';

@Injectable({
    providedIn: 'root'
})
export class SettingsService {
    private baseUrl = `${environment.apiHost}/api/settings`;

    constructor(private http: HttpClient) { }

    // GET /api/settings/export
    // This endpoint returns a CSV file, so we set the responseType to 'blob' to handle file downloads.
    exportUserData() {
        return this.http.get(`${this.baseUrl}/export`, { responseType: 'blob' });
    }

    // DELETE /api/settings/delete-account
    deleteAccount() {
        return this.http.delete(`${this.baseUrl}/delete-account`);
    }
}
