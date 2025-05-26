# Setup Instructions for ReelsRS

## Install Required Navigation Packages

Before running the app, you need to install the navigation packages. Run the following command in the project root directory:

```bash
npm install @react-navigation/native @react-navigation/native-stack react-native-screens react-native-safe-area-context axios
```

This will install:
- React Navigation (for screen navigation)
- React Native Screens (performance optimization for navigation)
- React Native Safe Area Context (for handling safe areas on different devices)
- Axios (for API calls)

## Running the App

After installing the packages, you can run the app using:

```bash
npx react-native run-android
```

## App Structure

The app includes:
1. Login Screen - For user authentication
2. Preferences Screen - For new users to select their interests
3. API Integration - Connected to your backend server

## Backend Integration Notes

The app is configured to connect to a backend server at `http://10.0.2.2:8000`.
This is the special IP that Android emulators use to access your computer's localhost.

If your backend is running on a different port or host, update the API_URL in:
`src/services/api.js` 