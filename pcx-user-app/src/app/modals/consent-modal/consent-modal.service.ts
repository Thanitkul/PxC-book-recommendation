import { Injectable } from '@angular/core';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import { ConsentModalComponent } from './consent-modal.component';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ConsentModalService {
    private dialogRef: MatDialogRef<ConsentModalComponent> | undefined;

    constructor(private dialog: MatDialog) { }

    /* 
     * Function to open the Genres modal.
     * args: none
     * return: Observable<any>
    */
    openModal(): Observable<any> {
        this.dialogRef = this.dialog.open(ConsentModalComponent, {
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
