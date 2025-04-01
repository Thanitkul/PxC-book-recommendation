import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment.development';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
    constructor(private http: HttpClient) { }
    signIn(body: object) {
        return this.http.post(`${environment.apiHost}/api/auth/signin`, body)
    }
    signUp(body: object) {
        return this.http.post(`${environment.apiHost}/api/auth/signup`, body)
    }
    getUserInfo() {
        return this.http.get(`${environment.apiHost}/api/auth/user-info`)
    }
    signout() {
        return this.http.get(`${environment.apiHost}/api/auth/signout`)
    }
}
