import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { SignInComponent } from './pages/sign-in/sign-in.component';
import { AdminComponent } from './pages/admin/admin.component';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { HTTP_INTERCEPTORS, HttpClientModule } from '@angular/common/http';
import { AddAdminModalComponent } from './modals/add-admin-modal/add-admin-modal.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatDialogModule } from '@angular/material/dialog';
import { VerifyInterceptor } from './core/auth/verify.interceptor';

@NgModule({
    declarations: [
        AppComponent,
        SignInComponent,
        AdminComponent,
        AddAdminModalComponent
    ],
    imports: [
        BrowserModule,
        AppRoutingModule,
        FormsModule,
        RouterModule,
        CommonModule,
        MatDialogModule,
        BrowserAnimationsModule,
        HttpClientModule
    ],
    providers: [{
        provide: HTTP_INTERCEPTORS,
        useClass: VerifyInterceptor,
        multi: true
    }],
    bootstrap: [AppComponent]
})
export class AppModule { }
