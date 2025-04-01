import { Injectable } from '@angular/core';
import { AddAdminModalComponent} from './add-admin-modal.component';
import { Observable } from 'rxjs';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';

@Injectable({
  providedIn: 'root'
})
export class AddAdminModalService {
  private dialogRef: MatDialogRef<AddAdminModalComponent> | undefined;

  constructor(private dialog: MatDialog) { }

  /* 
   * Function to open the Genres modal.
   * args: none
   * return: Observable<any>
  */
  openModal(): Observable<any> {
    this.dialogRef = this.dialog.open(AddAdminModalComponent, {
      panelClass: 'modal',
      minWidth: '90vh',
      hasBackdrop: true,
      disableClose: true,

    });

    return this.dialogRef.afterClosed();
  }

  closeModal(): void {
    this.dialog.closeAll();
  }
}
