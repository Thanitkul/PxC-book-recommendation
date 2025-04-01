import { Component } from '@angular/core';
import { MatDialogRef } from '@angular/material/dialog';
import { AddAdminModalService } from './add-admin-modal.service';

@Component({
  selector: 'app-add-admin-modal',
  templateUrl: './add-admin-modal.component.html',
  styleUrls: ['./add-admin-modal.component.css']
})
export class AddAdminModalComponent {

  ngOnInit(): void {
    if (localStorage.getItem('username') != null && localStorage.getItem('password') != null) {
      this.signinForm.username = localStorage.getItem('username')!
      this.signinForm.password = localStorage.getItem('password')!
    }
  }

  signinForm = {
    username: '',
    password: '',
  }
  constructor(
    public dialogRef: MatDialogRef<AddAdminModalComponent>,
    private consentModalServ: AddAdminModalService
  ) { }

  isUsernameEmtpy: boolean = false;
  isPasswordEmtpy: boolean = false;
  warningUsernameMessage: String = '';
  warningPasswordMessage: String = '';



  addAdmin() {
    console.log("added");
    this.doneButtonFunction();
  }

  /**
* Function to handle the "Done" button click.
* @return void
*/
  doneButtonFunction() {
     if (this.signinForm.username == '') {
            this.warningUsernameMessage = 'Please enter your username'
            this.isUsernameEmtpy = true;
        }
        if (this.signinForm.password == '') {
            this.warningPasswordMessage = 'Please enter your password'
            this.isPasswordEmtpy = true;
        }
      else {
      this.dialogRef.close({ username: this.signinForm.username, password: this.signinForm.password });
      }
  }
}

