import { Component, ViewChild, ElementRef, } from '@angular/core';
import { Router } from '@angular/router';
import { AddAdminModalService } from 'src/app/modals/add-admin-modal/add-admin-modal.service';
import { AuthService } from 'src/app/core/http/auth/auth.service';
import { AdminService } from 'src/app/core/http/admin/admin.service';

@Component({
    selector: 'app-admin',
    templateUrl: './admin.component.html',
    styleUrls: ['./admin.component.css']
})
export class AdminComponent {

    activeTab: string = 'book';
    searchQuery: string = '';
    selectedBook: any = null;
    selectedUser: any = null;
    filteredBooks: any[] = [];
    filteredUsers: any[] = [];

    // Book and User data
    books: any[] = [];
    users: any[] = [];

    logout() {
        this.authServ.signout().subscribe({
            next: (response: any) => {
                localStorage.removeItem('token');
                this.router.navigate(['/sign-in']);
            },
            error: (err: any) => {
                console.error('Logout error:', err);
            },
            complete: () => {
            }
        })
    }

    ngOnInit() {
        // Check if user is logged in
        this.authServ.getUserInfo().subscribe({
            next: (response: any) => {
                console.log('User info:', response);
            },
            error: (err: any) => {
                console.error('Error fetching user info:', err);
            }
        });
      this.adminService.listAllBooks().subscribe({
        next: (response: any) => {
          this.books = response.sort((a: any, b: any) => a.book_id - b.book_id);
          this.filteredBooks = [...this.books];
        },
        error: (err: any) => {
          console.error('Error fetching books:', err);
        }
      });

      this.adminService.listAllAdmins().subscribe({
        next: (response: any) => {
          this.users = response
          this.filteredUsers = [...this.users];
        },
        error: (err: any) => {
          console.error('Error fetching admins:', err);
        }
      });
    }

    navigateToLogin() {
        this.router.navigate(['/sign-in'])
    }

    constructor(private router: Router, private addAdminModalServe: AddAdminModalService, private authServ: AuthService, private adminService: AdminService) {
        this.filteredBooks = [...this.books];
        this.filteredUsers = [...this.users];
    }

    onBookRowClick(book: any) {
        this.selectedBook = book;
        console.log('Book selected:', book);
    }

    onUserRowClick(user: any) {
        this.selectedUser = user;
        console.log('User selected:', user);
    }

    // Change active tab
    setTab(tab: string): void {
        this.activeTab = tab;
        this.searchQuery = '';
    }

    onSearch(): void {
        const query = this.searchQuery.toLowerCase().trim();

        if (this.activeTab === 'book') {
            this.filteredBooks = this.books.filter(book =>
                book.title.toLowerCase().includes(query) ||
                book.author.toLowerCase().includes(query)
            );
        } else {
            this.filteredUsers = this.users.filter(user =>
                user.username.toLowerCase().includes(query)
            );
        }
    }

    deleteUser() {
      if (this.selectedUser) {
            const confirmDelete = window.confirm(`Are you sure you want to delete user "${this.selectedUser.username}"?`);

        if (confirmDelete) {
          this.adminService.deleteAdmin(this.selectedUser.id).subscribe({
            next: (response: any) => {
              this.users = this.users.filter(user => user.id !== this.selectedUser.id);
              this.filteredUsers = [...this.users];
              this.selectedUser = null;
            },
            error: (err: any) => {
              console.error('Error deleting user:', err);
            }
          });

          console.log('User deleted.');
        } else {
          console.log('User deletion canceled.');
        }
      }
    }

    addAdmin() {
      this.addAdminModalServe.openModal().subscribe({
        next: (data: any) => {
          this.router.navigate(['/'])
          if (data) {
            this.adminService.addAdmin(data.username, data.password).subscribe({
              next: (response: any) => {
                this.users.push({ id: response.id, username: response.username });
                this.filteredUsers = [...this.users];
              },
              error: (err: any) => {
                console.error('Error adding admin:', err);
                const errorMessage = err.error?.message ||
                  err.message ||
                  'Failed to add admin';
                window.alert(errorMessage);
              }
            });
          }
        },
        error: (err: any) => {
          alert("Failed to sign up")
        }
      })
    }

    isTableVisible = true;

    @ViewChild('tableContainer') tableContainer!: ElementRef;
    @ViewChild('hideButton') hideButton!: ElementRef;
    @ViewChild('showButton') showButton!: ElementRef;

    get isSelectedBookHidden(): boolean {
        return this.selectedBook?.is_visible === false;
    }

  toggleBookStatus() {
    if (this.selectedBook) {
      const updatedVisibility = !this.selectedBook.is_visible; 
      const book_id = this.selectedBook.book_id;
      this.adminService.updateBookVisiblity(book_id, updatedVisibility).subscribe({
        next: (response: any) => {
          const index = this.books.findIndex(b => b.book_id === book_id);
          if (index !== -1) {
            this.books[index].is_visible = updatedVisibility;
            this.selectedBook.is_visible = updatedVisibility;
            this.onSearch(); 
          }
        },
        error: (err: any) => {
          console.error('Error changing book status:', err);
        }
      });
    }
  }
}

