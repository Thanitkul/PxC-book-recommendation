import { Component } from '@angular/core';
import { MatDialogRef } from '@angular/material/dialog';
import { ConsentModalService } from './consent-modal.service';

@Component({
    selector: 'app-consent-modal',
    templateUrl: './consent-modal.component.html',
    styleUrls: ['./consent-modal.component.css']
})
export class ConsentModalComponent {
    constructor(
        public dialogRef: MatDialogRef<ConsentModalComponent>,
        private consentModalServ: ConsentModalService
    ) { }

    /**
 * Function to handle the "Done" button click.
 * @return void
 */
    doneButtonFunction() {
        this.dialogRef.close();

    }
}
