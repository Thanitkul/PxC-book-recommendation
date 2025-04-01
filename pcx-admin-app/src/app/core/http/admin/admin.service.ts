import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class AdminService {

  constructor(private http: HttpClient) { }

  listAllBooks() {
    return this.http.get(`${environment.apiHost}/api/admin/list-all-books`)
  }

  listAllAdmins() {
    return this.http.get(`${environment.apiHost}/api/admin/list-all-admins`)
  }

  updateBookVisiblity(book_id: number, is_visible: boolean) {
    return this.http.patch(`${environment.apiHost}/api/admin/update-book-visibility`, { book_id, is_visible })
  }

  addAdmin(username: string, password: string) {
    return this.http.post(`${environment.apiHost}/api/admin/add-admin`, { username, password })
  }

  deleteAdmin(id: number) {
    return this.http.delete(`${environment.apiHost}/api/admin/delete-admin`, { body: { id } })
  }
}
