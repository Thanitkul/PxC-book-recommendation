import { Injectable } from '@angular/core';
import { ActivatedRouteSnapshot, Router, RouterStateSnapshot, UrlTree } from '@angular/router';
import { Observable, of } from 'rxjs';
import { map, catchError } from 'rxjs/operators';
import { AuthService } from '../http/auth/auth.service';

@Injectable({
    providedIn: 'root'
})
export class AuthGuard {
    constructor(private router: Router, private authServ: AuthService) { }

    canActivate(
        route: ActivatedRouteSnapshot,
        state: RouterStateSnapshot
    ): Observable<boolean | UrlTree> {
        return this.authServ.getUserInfo().pipe(
            map(userInfo => {
                // If user information is retrieved, allow activation.
                // Adjust logic as needed.
                if (userInfo) {
                    return true;
                } else {
                    // Redirect if no user info
                    return this.router.createUrlTree(['/sign-in']);
                }
            }),
            catchError(err => {
                // In case of error, navigate to sign in
                return of(this.router.createUrlTree(['/sign-in']));
            })
        );
    }
}
