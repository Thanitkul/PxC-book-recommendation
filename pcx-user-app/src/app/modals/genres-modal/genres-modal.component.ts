import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { GenresModaService } from './genres-moda.service';
import { RecommendationService } from 'src/app/core/http/recommendation/recommendation.service';

@Component({
    selector: 'app-genres-modal',
    templateUrl: './genres-modal.component.html',
    styleUrls: ['./genres-modal.component.css']
})
export class GenresModalComponent {
    constructor(
        public dialogRef: MatDialogRef<GenresModalComponent>,
        private genresModalServ: GenresModaService,
        private recommendationServ: RecommendationService,
    ) { }

    ngOnInit() {
        this.getGenres();
    }

    genres = [];
    selectedGenres: string[] = [];

    getGenres() {
        // Check if user is logged in
        this.recommendationServ.getGenreChoices().subscribe({
            next: (response: any) => {
                this.genres = response.map((genre: string) =>
                    genre
                        .replace(/_/g, ' ')   // Replace underscores with spaces
                        .replace(/\//g, '/ ') // Add space after slashes
                );
            },
            error: (err: any) => {
                console.error('Error fetching user info:', err);
            }
        });
    }

    toggleGenre(genre: string) {
        const index = this.selectedGenres.indexOf(genre);
        if (index === -1) {
            this.selectedGenres.push(genre);
        } else {
            this.selectedGenres.splice(index, 1);
        }
    }


    /**
 * Function to handle the "Done" button click.
 * @return void
 */
    doneButtonFunction() {
        this.dialogRef.close(this.selectedGenres);

    }
}
