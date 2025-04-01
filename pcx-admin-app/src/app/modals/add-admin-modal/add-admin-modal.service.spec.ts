import { TestBed } from '@angular/core/testing';

import { AddAdminModalService } from './add-admin-modal.service';

describe('AddAdminModalService', () => {
  let service: AddAdminModalService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AddAdminModalService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
