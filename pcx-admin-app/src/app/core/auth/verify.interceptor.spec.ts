import { TestBed } from '@angular/core/testing';

import { VerifyInterceptor } from './verify.interceptor';

describe('VerifyInterceptor', () => {
  beforeEach(() => TestBed.configureTestingModule({
    providers: [
      VerifyInterceptor
      ]
  }));

  it('should be created', () => {
    const interceptor: VerifyInterceptor = TestBed.inject(VerifyInterceptor);
    expect(interceptor).toBeTruthy();
  });
});
