import axios from 'axios';

// Base URL for the backend API
// Note: For local development with Android emulator, use 10.0.2.2 instead of localhost
// This special IP address in the emulator is routed to your computer's localhost
const API_BASE_URL = 'http://10.0.2.2:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 seconds timeout
});

// Add request interceptor for debugging
api.interceptors.request.use(
  config => {
    console.log('Request:', config.method.toUpperCase(), config.url);
    if (config.data) {
      console.log('Request data:', JSON.stringify(config.data));
    }
    return config;
  },
  error => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for debugging
api.interceptors.response.use(
  response => {
    console.log('Response:', response.status, response.statusText);
    return response;
  },
  error => {
    console.error('Response error:', error.response?.status, error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Authentication related API calls
export const authAPI = {
  // Login function
  login: async (email, password) => {
    try {
      const response = await api.post('/login', { email, password });
      return response.data;
    } catch (error) {
      console.error('Login error:', error.response?.data || error.message);
      throw error;
    }
  },
  
  // Set initial preferences
  setInitialPreferences: async (userId, preferences) => {
    try {
      const response = await api.patch('/set-initial-preferences', {
        userId,
        initialPreferences: preferences
      });
      return response.data;
    } catch (error) {
      console.error('Set preferences error:', error.response?.data || error.message);
      throw error;
    }
  }
};

// Reels related API calls
export const reelsAPI = {
  // Get next similar reel
  getSimilarReels: async (userId, lastWatchedReelId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/reel?userId=${userId}&lastWatchedReelId=${lastWatchedReelId}`);
      const data = await response.json();
      console.log("Similar reel:", data.similar_reel_id);
      return data.similar_reel_id;
    } catch (error) {
      console.error('Error fetching similar reels:', error);
      throw error;
    }
  },

  // Get serendipity reel
  getSerendipityReel: async (userId, currentReelId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/serendipity-reel?userId=${userId}&currentReelId=${currentReelId}`);
      const data = await response.json();
      console.log("Serendipity reel:", data.reelId);
      return data.reelId;
    } catch (error) {
      console.error('Error fetching serendipity reel:', error);
      throw error;
    }
  },

  // Mark reel as watched
  markAsWatched: async (userId, reelId, rating = 3) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/mark-as-watched?userId=${userId}&reelId=${reelId}&rating=${rating}`, {
        method: 'PATCH',
      });
      return await response.json();
    } catch (error) {
      console.error('Error marking reel as watched:', error);
      throw error;
    }
  },

  // Get reel content (video)
  getReelContent: async (reelId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/reel-content?reelId=${reelId}`);
      return response;
    } catch (error) {
      console.error('Error fetching reel content:', error);
      throw error;
    }
  }
};

export default api; 