import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GenresModalComponent } from './genres-modal.component';

describe('GenresModalComponent', () => {
  let component: GenresModalComponent;
  let fixture: ComponentFixture<GenresModalComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [GenresModalComponent]
    });
    fixture = TestBed.createComponent(GenresModalComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
