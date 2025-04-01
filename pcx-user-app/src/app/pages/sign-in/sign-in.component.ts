import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from 'src/app/core/http/auth/auth.service';

@Component({
  selector: 'app-sign-in',
  templateUrl: './sign-in.component.html',
  styleUrls: ['./sign-in.component.css']
})
export class SignInComponent {
    ngOnInit(): void {
    }

    constructor(private router: Router, private authServ: AuthService) { }
    signinForm = {
        username: '',
        password: '',
    }
    isChecked: boolean = false;
    isUsernameEmtpy: boolean = false;
    isPasswordEmtpy: boolean = false;
    warningUsernameMessage: String = '';
    warningPasswordMessage: String = '';

    navigateToSignUp() {
        this.router.navigate(['/sign-up'])
    }

    submitForm() {
        if (this.signinForm.username == '') {
            this.warningUsernameMessage = 'Please enter your username'
            this.isUsernameEmtpy = true;
        }
        if (this.signinForm.password == '') {
            this.warningPasswordMessage = 'Please enter your password'
            this.isPasswordEmtpy = true;
        }
        if (this.signinForm.username != '' && this.signinForm.password != '') {
            this.isUsernameEmtpy = false;
            this.isPasswordEmtpy = false;
            this.warningUsernameMessage = '';
            this.warningPasswordMessage = '';

            this.authServ.signIn(this.signinForm).subscribe({
                next: (response: any) => {
                    localStorage.setItem('token', response["token"])
                },
                error: (err: any) => {
                    if (err["error"]["message"].includes("username")) {
                        this.warningUsernameMessage = "ชื่อผู้ใช้ไม่ถูกต้อง"
                        this.isUsernameEmtpy = true;
                    }
                    if (err["error"]["message"].includes("password")) {
                        this.warningPasswordMessage = "รหัสผ่านไม่ถูกต้อง"
                        this.isPasswordEmtpy = true;
                    }
                },
                complete: () => {
                    this.router.navigate(['/'])
                }
            })
        }
    }
}
