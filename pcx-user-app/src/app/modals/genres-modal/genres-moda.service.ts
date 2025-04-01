import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { GenresModalComponent } from './genres-modal.component';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
@Injectable({
  providedIn: 'root'
})
export class GenresModaService {
    private dialogRef: MatDialogRef<GenresModalComponent> | undefined;

    constructor(private dialog: MatDialog) { }

    /* 
     * Function to open the Genres modal.
     * args: none
     * return: Observable<any>
    */
    openModal(): Observable<any> {
        this.dialogRef = this.dialog.open(GenresModalComponent, {
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
