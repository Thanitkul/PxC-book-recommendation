import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { SignInComponent } from './pages/sign-in/sign-in.component';
import { RecommendationComponent } from './pages/recommendation/recommendation.component';
import { LibraryComponent } from './pages/library/library.component';
import { SignUpComponent } from './pages/sign-up/sign-up.component';
import { SettingsComponent } from './pages/settings/settings.component';
import { BrowseComponent } from './pages/browse/browse.component';
import { AuthGuard } from './core/auth/auth.guard';

const routes: Routes = [
    { path: '', redirectTo: 'recommendation', pathMatch: 'full' },
    { path: 'sign-in', component: SignInComponent },
    { path: 'sign-up', component: SignUpComponent },
    { path: 'settings', component: SettingsComponent, canActivate: [AuthGuard]},
    { path: 'browse', component: BrowseComponent, canActivate: [AuthGuard]},
    { path: 'recommendation', component: RecommendationComponent, canActivate: [AuthGuard]},
    { path: 'library', component: LibraryComponent, canActivate: [AuthGuard]},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
