import { Injectable, signal } from '@angular/core';
import { environment } from '../environments/environment';

export interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  organization?: string;
}

export interface AuthResponse {
  message: string;
  access_token: string;
  refresh_token: string;
  user: User;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  isAuthenticated = signal<boolean>(true);
  currentUser = signal<User | null>({
    id: 1,
    username: 'guest',
    email: 'guest@example.com',
    full_name: 'Guest User',
    organization: 'Public'
  });

  constructor() {}

  async register(username: string, email: string, password: string): Promise<{success: boolean, message: string}> {
    return { success: true, message: 'Registration disabled (public access only)' };
  }

  async login(username: string, password: string): Promise<{success: boolean, message: string}> {
    return { success: true, message: 'Login disabled (public access only)' };
  }

  logout(): void {
    // No-op
  }

  getToken(): string | null {
    return 'dummy-token';
  }

  async authenticatedFetch(endpoint: string, options: RequestInit = {}): Promise<Response> {
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };

    return fetch(`${environment.apiUrl}${endpoint}`, {
      ...options,
      headers
    });
  }
}
