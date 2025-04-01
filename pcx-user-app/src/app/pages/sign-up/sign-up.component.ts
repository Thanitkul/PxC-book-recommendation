import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from 'src/app/core/http/auth/auth.service';
import { ConsentModalService } from 'src/app/modals/consent-modal/consent-modal.service';

@Component({
    selector: 'app-sign-up',
    templateUrl: './sign-up.component.html',
    styleUrls: ['./sign-up.component.css']
})
export class SignUpComponent {


    constructor(private router: Router, private consentModalServe: ConsentModalService, private authServ: AuthService) { }
    signinForm = {
        username: '',
        password: '',
        confirmPassword: ''
    };

    isChecked: boolean = false;
    formSubmitted: boolean = false;

    isPasswordEmpty: boolean = false;
    isUsernameEmpty: boolean = false;
    isConfirmPasswordEmpty: boolean = false;

    isPasswordInvalid: boolean = false;
    warningUsernameMessage: string = '';
    warningPasswordMessage: string = '';
    warningConfirmPasswordMessage: string = '';

    ngOnInit(): void {
        const storedUsername = localStorage.getItem('username');
        const storedPassword = localStorage.getItem('password');

        if (storedUsername && storedPassword) {
            this.signinForm.username = storedUsername;
            this.signinForm.password = storedPassword;
            this.isChecked = true;
        }
    }

    isPasswordValid(password: string): boolean {
        const pattern = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&_. ,])[A-Za-z\d@$!%*?&_. ,]{6,}$/;
        return pattern.test(password);
    }

    validatePassword(): void {
        const password = this.signinForm.password;

        if (!password && this.formSubmitted) {
            this.warningPasswordMessage = 'Please enter your password';
            this.isPasswordInvalid = false;
        } else if (password && !this.isPasswordValid(password)) {
            this.warningPasswordMessage =
                'Password must be at least 6 characters, include uppercase, lowercase, number, and special character.';
            this.isPasswordInvalid = true;
        } else {
            this.warningPasswordMessage = '';
            this.isPasswordInvalid = false;
        }

        this.validateConfirmPassword();
    }

    validateConfirmPassword(): void {
        const { password, confirmPassword } = this.signinForm;

        if (!confirmPassword && this.formSubmitted) {
            this.warningConfirmPasswordMessage = 'Please confirm your password';
        } else if (password !== confirmPassword && confirmPassword) {
            this.warningConfirmPasswordMessage = 'Passwords do not match';
        } else {
            this.warningConfirmPasswordMessage = '';
        }
    }

    validateUsername(): void {
        const username = this.signinForm.username.trim();
        if (!username && this.formSubmitted) {
            this.warningUsernameMessage = 'Please enter your username';
        } else {
            this.warningUsernameMessage = '';
        }

    }

    navigateToSignIn(): void {
        this.router.navigate(['/sign-in']);
    }


    validateAllFields(): void {
        this.validateUsername();
        this.validatePassword();
        this.validateConfirmPassword();
    }

    submitForm(): void {
        this.formSubmitted = true;
        this.validateAllFields();

        const isEmpty =
            !this.signinForm.username.trim() ||
            !this.signinForm.password ||
            !this.signinForm.confirmPassword;

        const hasError =
            isEmpty || this.isPasswordInvalid || this.warningConfirmPasswordMessage !== '';

        if (!hasError) {
            this.consentModalServe.openModal().subscribe({
                next: (response: any) => {
                    this.authServ.signUp(this.signinForm).subscribe({
                        next: (response: any) => {
                            localStorage.setItem('token', response["token"])
                        },
                        error: (err: any) => {
                            if (err["error"]["message"].includes("username")) {
                                this.warningUsernameMessage = "ชื่อผู้ใช้ไม่ถูกต้อง"
                            }
                            if (err["error"]["message"].includes("password")) {
                                this.warningPasswordMessage = "รหัสผ่านไม่ถูกต้อง"
                            }
                        },
                        complete: () => {
                            this.router.navigate(['/'])
                        }
                    })
                },
                error: (err: any) => {
                    alert("Failed to sign up")
                }
            })
        }
    }
}
