import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ConsentModalComponent } from './consent-modal.component';

describe('ConsentModalComponent', () => {
  let component: ConsentModalComponent;
  let fixture: ComponentFixture<ConsentModalComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [ConsentModalComponent]
    });
    fixture = TestBed.createComponent(ConsentModalComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
