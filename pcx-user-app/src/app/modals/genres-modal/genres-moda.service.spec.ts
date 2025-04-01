import { TestBed } from '@angular/core/testing';

import { GenresModaService } from '../genres-moda.service';

describe('GenresModaService', () => {
  let service: GenresModaService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(GenresModaService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
