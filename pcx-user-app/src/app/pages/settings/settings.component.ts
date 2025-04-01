import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from 'src/app/core/http/auth/auth.service';
import { SettingsService } from 'src/app/core/http/settings/settings.service';

@Component({
    selector: 'app-settings',
    templateUrl: './settings.component.html',
    styleUrls: ['./settings.component.css']
})
export class SettingsComponent {

    constructor(
        private router: Router,
        private authServ: AuthService,
        private settingsService: SettingsService
    ) { }

    navigateToLibrary() {
        this.router.navigate(['/library']);
    }

    navigateToRecommend() {
        this.router.navigate(['/recommendation']);
    }

    navigateToBrowse() {
        this.router.navigate(['/browse']);
    }

    navigateToSignIn() {
        this.router.navigate(['/sign-in']);
    }

    onExport() {
        // Call the service to export user data
        this.settingsService.exportUserData().subscribe({
            next: (data: Blob) => {
                // Create a blob from the CSV data and generate a URL for it
                const blob = new Blob([data], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                // Create an anchor element and trigger a download
                const a = document.createElement('a');
                a.href = url;
                a.download = 'user_data.csv';
                a.click();
                // Clean up the object URL
                window.URL.revokeObjectURL(url);
            },
            error: (err: any) => {
                console.error('Error exporting user data:', err);
                alert('Error exporting user data.');
            }
        });
    }

    onDelete() {
        // Confirm before deleting the account
        if (confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
            this.settingsService.deleteAccount().subscribe({
                next: (response: any) => {
                    alert('Account deleted successfully.');
                    this.router.navigate(['/sign-in']);
                },
                error: (err: any) => {
                    console.error('Error deleting account:', err);
                    alert('Error deleting account.');
                }
            });
        }
    }

    onLogout() {
        this.authServ.signout().subscribe({
            next: (response: any) => {
                localStorage.removeItem('token');
                this.router.navigate(['/sign-in']);
            },
            error: (err: any) => {
                console.error('Logout error:', err);
            }
        });
    }
}
